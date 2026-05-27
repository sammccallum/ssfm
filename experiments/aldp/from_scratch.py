import math
import os
import pickle

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from jaxtyping import Array, Float, PRNGKeyArray
from matplotlib.colors import LogNorm
from scoremd.data.dataset.aldp import ALDPDataset, CoarseGrainingLevel
from scoremd.utils.evaluation import phi_psi_metrics

from ssfm.diffusion import VPDiffusion
from ssfm.flow_map import EulerMaruyamaFlowMap
from ssfm.graph_transformer import GraphTransformerEMStepModel
from ssfm.losses import (
    UncertaintyDistillationLoss,
    UncertaintyJointLoss,
    UncertaintyMLP,
    UncertaintyScoreLoss,
)
from ssfm.typing import Y


def ema_update(
    model: EulerMaruyamaFlowMap,
    ema_model: EulerMaruyamaFlowMap,
    decay: float,
) -> EulerMaruyamaFlowMap:
    params, static = eqx.partition(model, eqx.is_array)
    ema_params, _ = eqx.partition(ema_model, eqx.is_array)
    new_ema_params = jax.tree.map(
        lambda e, p: decay * e + (1 - decay) * p, ema_params, params
    )
    return eqx.combine(new_ema_params, static)


def _random_rotation_matrices(
    key: PRNGKeyArray, batch_size: int
) -> Float[Array, "batch 3 3"]:
    subkeys = jax.random.split(key, batch_size)

    def one(k):
        q = jax.random.normal(k, (4,))
        q = q / jnp.linalg.norm(q)
        w, x, y, z = q
        return jnp.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

    return jax.vmap(one)(subkeys)


def apply_random_rotations(
    batch: Float[Array, "batch d"],
    n_atoms: int,
    key: PRNGKeyArray,
) -> Float[Array, "batch d"]:
    orig_shape = batch.shape
    x = batch.reshape(batch.shape[0], n_atoms, 3)
    R = _random_rotation_matrices(key, batch.shape[0])
    offset = jnp.mean(x, axis=1, keepdims=True)
    x_rot = jnp.einsum("bij,bnj->bni", R, x - offset) + offset
    return x_rot.reshape(orig_shape)


@eqx.filter_jit
def train_step(
    flow_map: EulerMaruyamaFlowMap,
    ema_flow_map: EulerMaruyamaFlowMap,
    y0_batch: Float[Array, "batch d"],
    n_atoms: int,
    invariance: bool,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    joint_loss: UncertaintyJointLoss,
    ema_decay: float,
    key: PRNGKeyArray,
):
    key_rot, key_loss = jax.random.split(key)
    if invariance:
        y0_batch = apply_random_rotations(y0_batch, n_atoms, key_rot)
    loss, fm_grads, u_grads = joint_loss.value_and_grad(
        flow_map, ema_flow_map, y0_batch, key_loss
    )
    updates, opt_state = optimizer.update(
        (fm_grads, u_grads), opt_state, (flow_map, joint_loss.u_mlp)
    )
    fm_updates, u_updates = updates
    flow_map = eqx.apply_updates(flow_map, fm_updates)
    new_u_mlp = eqx.apply_updates(joint_loss.u_mlp, u_updates)
    joint_loss = eqx.tree_at(lambda l: l.u_mlp, joint_loss, new_u_mlp)
    ema_flow_map = ema_update(flow_map, ema_flow_map, ema_decay)
    return flow_map, ema_flow_map, opt_state, joint_loss, loss


def sample_flow_map(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key: PRNGKeyArray,
    n_samples: int,
    data_dim: int,
    step_size: float,
    n_steps: int,
) -> Float[Array, "batch d"]:
    flow_map = eqx.nn.inference_mode(flow_map)
    time_grid = jnp.linspace(1.0, t_eps, n_steps + 1)

    def sample_one(k: PRNGKeyArray) -> Y:
        k_init, k_vbt = jax.random.split(k)
        y = jax.random.normal(k_init, (data_dim,))
        vbt = diffrax.VirtualBrownianTree(
            t0=t_eps,
            t1=1.0,
            tol=step_size / 4,
            shape=(data_dim,),
            key=k_vbt,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )

        def scan_fn(y, i):
            s = jnp.clip(time_grid[i], t_eps, 1.0)
            t = jnp.clip(time_grid[i + 1], t_eps, 1.0)
            levy = vbt.evaluate(s, t, use_levy=True)
            return flow_map(y, s, t, levy.W, levy.H, levy.K), None

        y, _ = jax.lax.scan(scan_fn, y, jnp.arange(n_steps))
        return y

    keys = jax.random.split(key, n_samples)
    return jax.vmap(sample_one)(keys)


def compute_pmf(
    samples_phys: Float[Array, "batch d"],
    target_phi: np.ndarray,
    target_psi: np.ndarray,
    dataset: ALDPDataset,
    n_bins: int = 64,
) -> tuple[float, float]:
    sampled_phi, sampled_psi = dataset.get_2d_features(samples_phys)
    rms_fe_sq, rms_mjs = phi_psi_metrics(
        target_phi,
        target_psi,
        np.asarray(sampled_phi),
        np.asarray(sampled_psi),
        n_bins=n_bins,
    )
    return float(rms_fe_sq), float(rms_mjs)


def _plot_phi_psi(
    ax: plt.Axes,
    samples_phys: Float[Array, "n d"],
    dataset: ALDPDataset,
    title: str,
    bins: int = 100,
) -> None:
    phi, psi = dataset.get_2d_features(samples_phys)
    ax.hist2d(
        np.asarray(phi),
        np.asarray(psi),
        bins=bins,
        range=[[-math.pi, math.pi], [-math.pi, math.pi]],
        norm=LogNorm(),
        rasterized=True,
    )
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(r"$\psi$")
    ax.set_title(title)


def make_sample_grid(
    ema_flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key: PRNGKeyArray,
    dataset: ALDPDataset,
    data_dim: int,
    step_sizes: list[float],
    norm_factor: float,
    n_samples: int = 2000,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(step_sizes), figsize=(4 * len(step_sizes), 4))
    if len(step_sizes) == 1:
        axes = [axes]

    for ax, ss in zip(axes, step_sizes):
        key, k = jax.random.split(key)
        n_steps = max(1, math.ceil((1.0 - t_eps) / ss))
        samples = sample_flow_map(
            ema_flow_map, t_eps, k, n_samples, data_dim, ss, n_steps
        )
        _plot_phi_psi(ax, samples / norm_factor, dataset, f"step_size={ss}", bins=60)

    fig.tight_layout()
    return fig


def plot_samples_by_step_size(
    samples_by_ss: dict[float, Float[Array, "n d"]],
    dataset: ALDPDataset,
    norm_factor: float,
    bins: int = 100,
) -> plt.Figure:
    step_sizes = list(samples_by_ss.keys())
    fig, axes = plt.subplots(1, len(step_sizes), figsize=(4 * len(step_sizes), 4))
    if len(step_sizes) == 1:
        axes = [axes]

    for ax, ss in zip(axes, step_sizes):
        _plot_phi_psi(
            ax,
            samples_by_ss[ss] / norm_factor,
            dataset,
            f"step_size={ss}",
            bins=bins,
        )

    fig.tight_layout()
    return fig


def save_checkpoint(
    path: str,
    flow_map: EulerMaruyamaFlowMap,
    ema_flow_map: EulerMaruyamaFlowMap,
    opt_state: optax.OptState,
    u_mlp: UncertaintyMLP,
    step: int,
    key: PRNGKeyArray,
):
    os.makedirs(path, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(path, "flow_map.eqx"), flow_map)
    eqx.tree_serialise_leaves(os.path.join(path, "ema_flow_map.eqx"), ema_flow_map)
    eqx.tree_serialise_leaves(os.path.join(path, "u_mlp.eqx"), u_mlp)
    with open(os.path.join(path, "opt_state.pkl"), "wb") as f:
        pickle.dump(opt_state, f)
    with open(os.path.join(path, "train_state.pkl"), "wb") as f:
        pickle.dump({"step": step, "key": key}, f)
    print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(
    path: str,
    flow_map: EulerMaruyamaFlowMap,
    ema_flow_map: EulerMaruyamaFlowMap,
    u_mlp: UncertaintyMLP,
):
    flow_map = eqx.tree_deserialise_leaves(os.path.join(path, "flow_map.eqx"), flow_map)
    ema_flow_map = eqx.tree_deserialise_leaves(
        os.path.join(path, "ema_flow_map.eqx"), ema_flow_map
    )
    u_mlp = eqx.tree_deserialise_leaves(os.path.join(path, "u_mlp.eqx"), u_mlp)
    with open(os.path.join(path, "opt_state.pkl"), "rb") as f:
        opt_state = pickle.load(f)
    with open(os.path.join(path, "train_state.pkl"), "rb") as f:
        train_state = pickle.load(f)
    print(f"Resumed from checkpoint at step {train_state['step']} from {path}")
    return (
        flow_map,
        ema_flow_map,
        opt_state,
        u_mlp,
        train_state["step"],
        train_state["key"],
    )


def main():
    batch_size = 1024
    n_train_steps = 400_000
    lr = 1e-3
    eta = 0.75
    ema_decay = 0.999
    dt = 1e-3
    h_max = 0.52
    t_eps = 1e-5
    reverse_eta = 1.0
    beta_min = 0.1
    beta_max = 20.0
    hidden_nf = 96
    n_layers = 3
    heads = 8
    dim_head = 64
    ff_mult = 4
    time_dim = 64
    dropout_rate = 0.0
    fourier_dim = 64
    log_every = 1000
    sample_every = 10_000
    checkpoint_every = 50_000
    checkpoint_dir = "experiments/aldp/checkpoints/v1_from_scratch"
    sample_step_sizes = [0.01, 0.05, 0.1, 0.25, 0.5]
    n_eval_samples = 4_096
    warmup_steps = 1_000
    min_lr = 1e-5
    invariance = False

    config = {
        "batch_size": batch_size,
        "n_train_steps": n_train_steps,
        "lr": lr,
        "eta": eta,
        "ema_decay": ema_decay,
        "dt": dt,
        "h_max": h_max,
        "t_eps": t_eps,
        "reverse_eta": reverse_eta,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "hidden_nf": hidden_nf,
        "n_layers": n_layers,
        "heads": heads,
        "dim_head": dim_head,
        "ff_mult": ff_mult,
        "time_dim": time_dim,
        "dropout_rate": dropout_rate,
        "fourier_dim": fourier_dim,
        "sample_step_sizes": sample_step_sizes,
        "n_eval_samples": n_eval_samples,
        "warmup_steps": warmup_steps,
        "min_lr": min_lr,
        "invariance": invariance,
    }

    wandb.init(project="Stochastic Flow Map", config=config, name="md-aldp-v1-scratch")

    key = jax.random.PRNGKey(0)
    key, key_model, key_u = jax.random.split(key, 3)

    print("Loading ALDP dataset...")
    dataset = ALDPDataset(
        coarse_graining_level=CoarseGrainingLevel.FULL,
        limit_samples=50_000,
        validation=False,
        seed=0,
    )
    data_dim = dataset.train.data.shape[1]
    n_atoms = data_dim // 3
    assert data_dim == n_atoms * 3, f"data_dim {data_dim} not divisible by 3"
    print(
        f"  data_dim = {data_dim}, n_atoms = {n_atoms}, "
        f"n_samples = {dataset.train.data.shape[0]}"
    )

    y_raw = np.asarray(dataset.train.data)
    if invariance:
        y_shaped = y_raw.reshape(-1, n_atoms, 3)
        y_shaped = y_shaped - y_shaped.mean(axis=1, keepdims=True)
        y_raw = y_shaped.reshape(-1, data_dim)

    norm_factor = 1.0 / dataset.std
    y_norm = jnp.asarray(y_raw) * norm_factor
    target_phi, target_psi = dataset.get_2d_features(dataset.train.data)
    target_phi = np.asarray(target_phi)
    target_psi = np.asarray(target_psi)

    diffusion = VPDiffusion(
        beta_min=beta_min, beta_max=beta_max, reverse_eta=reverse_eta
    )
    step_model = GraphTransformerEMStepModel(
        n_atoms=n_atoms,
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        heads=heads,
        dim_head=dim_head,
        ff_mult=ff_mult,
        time_dim=time_dim,
        dropout_rate=dropout_rate,
        key=key_model,
    )
    flow_map = EulerMaruyamaFlowMap(step_model=step_model)
    ema_flow_map = flow_map

    score_loss = UncertaintyScoreLoss(diffusion=diffusion, dt=dt, t_eps=t_eps)
    distill_loss = UncertaintyDistillationLoss(
        diffusion=diffusion, dt=dt, h_max=h_max, t_eps=t_eps
    )
    u_mlp = UncertaintyMLP(fourier_dim=fourier_dim, key=key_u)
    joint_loss = UncertaintyJointLoss(
        score_loss=score_loss, distill_loss=distill_loss, u_mlp=u_mlp, eta=eta
    )

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(flow_map, eqx.is_array)))
    print(f"Flow map params: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=n_train_steps,
        end_value=min_lr,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(schedule),
    )
    opt_state = optimizer.init(
        (
            eqx.filter(flow_map, eqx.is_array),
            eqx.filter(joint_loss.u_mlp, eqx.is_array),
        )
    )

    for step in range(n_train_steps):
        key, key_batch, key_step = jax.random.split(key, 3)
        idx = jax.random.randint(key_batch, (batch_size,), 0, y_norm.shape[0])
        y0_batch = y_norm[idx]

        flow_map, ema_flow_map, opt_state, joint_loss, loss = train_step(
            flow_map,
            ema_flow_map,
            y0_batch,
            n_atoms,
            invariance,
            opt_state,
            optimizer,
            joint_loss,
            ema_decay,
            key_step,
        )

        if step % log_every == 0:
            loss_val = float(loss)
            print(f"Step {step:>6d} | Loss: {loss_val:.4f}")
            wandb.log({"loss": loss_val}, step=step)

        if step > 0 and step % sample_every == 0:
            key, key_samples = jax.random.split(key)
            for ss in sample_step_sizes:
                key_samples, k = jax.random.split(key_samples)
                n_steps = max(1, math.ceil((1.0 - t_eps) / ss))
                samples = sample_flow_map(
                    ema_flow_map, t_eps, k, n_eval_samples, data_dim, ss, n_steps
                )
                pmf, mjs = compute_pmf(
                    samples / norm_factor, target_phi, target_psi, dataset
                )
                wandb.log(
                    {f"pmf/step_size_{ss}": pmf, f"mjs/step_size_{ss}": mjs},
                    step=step,
                )

            key, key_plot = jax.random.split(key)
            fig = make_sample_grid(
                ema_flow_map,
                t_eps,
                key_plot,
                dataset,
                data_dim,
                sample_step_sizes,
                norm_factor,
            )
            wandb.log({"samples": wandb.Image(fig)}, step=step)
            plt.close(fig)

        if step > 0 and step % checkpoint_every == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, "latest"),
                flow_map,
                ema_flow_map,
                opt_state,
                joint_loss.u_mlp,
                step,
                key,
            )

    os.makedirs("experiments/aldp/models", exist_ok=True)
    eqx.tree_serialise_leaves(
        "experiments/aldp/models/aldp_v1_from_scratch.eqx", ema_flow_map
    )
    print("Saved experiments/aldp/models/aldp_v1_from_scratch.eqx")

    n_final_samples = 120_000
    final_sample_batch_size = 4_000
    assert n_final_samples % final_sample_batch_size == 0, (
        f"n_final_samples ({n_final_samples}) must be divisible by "
        f"final_sample_batch_size ({final_sample_batch_size})"
    )
    n_final_batches = n_final_samples // final_sample_batch_size
    key, key_final = jax.random.split(key)
    final_samples_by_ss: dict[float, jnp.ndarray] = {}
    for ss in sample_step_sizes:
        key_final, k = jax.random.split(key_final)
        n_steps = max(1, math.ceil((1.0 - t_eps) / ss))
        batch_keys = jax.random.split(k, n_final_batches)
        sample_batches = [
            sample_flow_map(
                ema_flow_map,
                t_eps,
                bk,
                final_sample_batch_size,
                data_dim,
                ss,
                n_steps,
            )
            for bk in batch_keys
        ]
        samples = jnp.concatenate(sample_batches, axis=0)
        final_samples_by_ss[ss] = samples
        pmf, mjs = compute_pmf(samples / norm_factor, target_phi, target_psi, dataset)
        print(f"Final | step_size={ss} | PMF={pmf:.4f} | MJS={mjs:.4f}")
        wandb.log(
            {f"pmf/step_size_{ss}": pmf, f"mjs/step_size_{ss}": mjs},
            step=n_train_steps,
        )

    fig = plot_samples_by_step_size(final_samples_by_ss, dataset, norm_factor)
    fig.savefig("experiments/aldp/aldp_v1_from_scratch_samples.png", dpi=150)
    wandb.log({"samples": wandb.Image(fig)}, step=n_train_steps)
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    main()
