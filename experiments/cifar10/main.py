import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")
import math
import pickle
import tarfile
import urllib.request

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from ssfm.diffusion import VPDiffusion
from ssfm.edm2_unet import EDM2_UNet, EDM2EMStepModel
from ssfm.flow_map import EulerMaruyamaFlowMap
from ssfm.losses import (
    UncertaintyDistillationLoss,
    UncertaintyJointLoss,
    UncertaintyMLP,
    UncertaintyScoreLoss,
)


def cifar10(path="experiments/cifar10/data"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(path, "cifar-10-python.tar.gz")

    os.makedirs(path, exist_ok=True)
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, tar_path)

    all_images = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for i in range(1, 6):
            member = tar.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            batch = pickle.load(member, encoding="bytes")  # pyright: ignore
            all_images.append(batch[b"data"])

    images = np.concatenate(all_images, axis=0)
    images = jnp.array(images, dtype=jnp.float32) / 255.0

    data_mean = jnp.mean(images, axis=0)
    data_std = jnp.clip(jnp.std(images, axis=0), 1e-6)
    images = (images - data_mean) / data_std
    return images, data_mean, data_std


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


@eqx.filter_jit
def train_step(
    flow_map: EulerMaruyamaFlowMap,
    ema_flow_map: EulerMaruyamaFlowMap,
    y0_batch: Float[Array, "batch 3 32 32"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    joint_loss: UncertaintyJointLoss,
    ema_decay: float,
    key: PRNGKeyArray,
):
    loss, fm_grads, u_grads = joint_loss.value_and_grad(
        flow_map, ema_flow_map, y0_batch, key
    )

    updates, opt_state = optimizer.update(
        (fm_grads, u_grads),
        opt_state,
        (flow_map, joint_loss.u_mlp),  # pyright: ignore
    )
    fm_updates, u_updates = updates  # pyright: ignore
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
    data_shape: tuple[int, ...],
    step_size: float,
    n_steps: int,
):
    flow_map = eqx.nn.inference_mode(flow_map)

    time_grid = jnp.linspace(1.0, t_eps, n_steps + 1)

    def sample_one(key):
        key_init, key_vbt = jax.random.split(key)
        y = jax.random.normal(key_init, data_shape)

        vbt = diffrax.VirtualBrownianTree(
            t0=t_eps,
            t1=1.0,
            tol=step_size / 4,
            shape=data_shape,
            key=key_vbt,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )

        def scan_fn(y, i):
            s = jnp.clip(time_grid[i], t_eps, 1.0)
            t = jnp.clip(time_grid[i + 1], t_eps, 1.0)
            levy = vbt.evaluate(s, t, use_levy=True)
            y = flow_map(y, s, t, levy.W, levy.H, levy.K)  # pyright: ignore
            return y, None

        y, _ = jax.lax.scan(scan_fn, y, jnp.arange(n_steps))
        return y

    keys = jax.random.split(key, n_samples)
    return jax.vmap(sample_one)(keys)


def sample_flow_map_batched(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key: PRNGKeyArray,
    n_samples: int,
    data_shape: tuple[int, ...],
    step_size: float,
    n_steps: int,
    batch_size: int = 256,
):
    all_samples = []
    remaining = n_samples
    while remaining > 0:
        key, key_batch = jax.random.split(key)
        bs = min(batch_size, remaining)
        batch = sample_flow_map(
            flow_map, t_eps, key_batch, bs, data_shape, step_size, n_steps
        )
        all_samples.append(batch)
        remaining -= bs
    return jnp.concatenate(all_samples, axis=0)


def make_sample_grid(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key: PRNGKeyArray,
    data_shape: tuple[int, ...],
    step_sizes: list[float],
    data_mean: jnp.ndarray,
    data_std: jnp.ndarray,
    n_cols: int = 8,
):
    n_rows = len(step_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, ss in enumerate(step_sizes):
        key, key_sample = jax.random.split(key)
        n_steps = max(1, math.ceil((1.0 - t_eps) / ss))
        samples = sample_flow_map(
            flow_map, t_eps, key_sample, n_cols, data_shape, ss, n_steps
        )
        samples = samples * data_std[None] + data_mean[None]
        samples = jnp.clip(samples, 0.0, 1.0)

        for col in range(n_cols):
            ax = axes[row, col]
            img = np.array(samples[col].transpose(1, 2, 0))
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"h={ss}", fontsize=10)

    fig.tight_layout()
    return fig


def format_step(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def parse_step(name: str) -> int | None:
    if name.endswith("k"):
        try:
            return int(name[:-1]) * 1000
        except ValueError:
            return None
    try:
        return int(name)
    except ValueError:
        return None


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None
    best_step = -1
    best_path = None
    for name in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(full):
            continue
        step = parse_step(name)
        if step is None:
            continue
        if step > best_step:
            best_step = step
            best_path = full
    return best_path


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
    optimizer: optax.GradientTransformation,
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
    in_channels = 3
    resolution = 32
    data_shape = (in_channels, resolution, resolution)
    batch_size = 512
    n_train_steps = 400_000
    lr = 1e-3
    eta = 0.75
    ema_decay = 0.999
    dt = 0.01
    h_max = 0.52
    t_eps = 1e-5
    beta_min = 0.1
    beta_max = 20.0
    sample_every = 10_000
    checkpoint_every = 50_000
    checkpoint_dir = "experiments/cifar10/checkpoints"
    warmup_steps = 5_000
    sample_step_sizes = [1 / 16, 1 / 8, 1 / 4, 1 / 2]

    base_channels = 128
    channel_mult = (2, 2, 2)
    num_groups = 8
    dropout_rate = 0.13
    attn_resolutions = [16]
    head_dim = 64
    num_res_blocks = 4

    diff_base_channels = 64
    diff_channel_mult = (2, 2, 2)
    diff_attn_resolutions = [16]

    config = {
        "architecture": "edm2",
        "dataset": "cifar10",
        "batch_size": batch_size,
        "n_train_steps": n_train_steps,
        "lr": lr,
        "eta": eta,
        "ema_decay": ema_decay,
        "dt": dt,
        "h_max": h_max,
        "t_eps": t_eps,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "sample_every": sample_every,
        "checkpoint_every": checkpoint_every,
        "warmup_steps": warmup_steps,
        "sample_step_sizes": sample_step_sizes,
        "base_channels": base_channels,
        "channel_mult": channel_mult,
        "attn_resolutions": attn_resolutions,
        "diff_base_channels": diff_base_channels,
        "diff_channel_mult": diff_channel_mult,
        "diff_attn_resolutions": diff_attn_resolutions,
        "num_res_blocks": num_res_blocks,
        "dropout_rate": dropout_rate,
    }

    wandb.init(project="Stochastic Flow Map", config=config)

    devices = jax.devices()
    n_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch", None, None, None))
    replicate_sharding = NamedSharding(mesh, P())
    assert batch_size % n_devices == 0, (
        f"batch_size {batch_size} not divisible by {n_devices} devices"
    )
    print(f"Training on {n_devices} devices: {devices}")

    key = jax.random.PRNGKey(0)
    key, key_drift, key_diff, key_u = jax.random.split(key, 4)

    dataset, data_mean, data_std = cifar10()
    dataset = dataset.reshape(-1, *data_shape)
    data_mean = data_mean.reshape(data_shape)
    data_std = data_std.reshape(data_shape)
    print(f"Dataset: {dataset.shape}")

    diffusion = VPDiffusion(reverse_eta=1.0, beta_min=beta_min, beta_max=beta_max)

    drift_net = EDM2_UNet(
        in_channels=4 * in_channels,
        out_channels=in_channels,
        resolution=resolution,
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_groups=num_groups,
        num_res_blocks=num_res_blocks,
        dropout_rate=dropout_rate,
        attn_resolutions=attn_resolutions,
        head_dim=head_dim,
        key=key_drift,
    )

    diffusion_net = EDM2_UNet(
        in_channels=3 * in_channels,
        out_channels=in_channels,
        resolution=resolution,
        base_channels=diff_base_channels,
        channel_mult=diff_channel_mult,
        num_groups=num_groups,
        num_res_blocks=num_res_blocks,
        dropout_rate=0.0,
        attn_resolutions=diff_attn_resolutions,
        head_dim=head_dim,
        key=key_diff,
    )

    step_model = EDM2EMStepModel(
        drift_net=drift_net,
        diffusion_net=diffusion_net,
    )
    flow_map = EulerMaruyamaFlowMap(step_model=step_model)
    ema_flow_map = flow_map

    n_drift = sum(x.size for x in jax.tree.leaves(eqx.filter(drift_net, eqx.is_array)))
    n_diff = sum(
        x.size for x in jax.tree.leaves(eqx.filter(diffusion_net, eqx.is_array))
    )
    print(f"Drift EDM2 UNet params: {n_drift:,}")
    print(f"Diffusion EDM2 UNet params: {n_diff:,}")
    print(f"Total params: {n_drift + n_diff:,}")
    wandb.config.update({"n_drift_params": n_drift, "n_diff_params": n_diff})

    score_loss = UncertaintyScoreLoss(diffusion=diffusion, dt=dt, t_eps=t_eps)
    distill_loss = UncertaintyDistillationLoss(
        diffusion=diffusion, dt=dt, h_max=h_max, t_eps=t_eps
    )
    u_mlp = UncertaintyMLP(fourier_dim=128, key=key_u)
    joint_loss = UncertaintyJointLoss(
        score_loss=score_loss, distill_loss=distill_loss, u_mlp=u_mlp, eta=eta
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=n_train_steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule, b2=0.99),
    )
    opt_state = optimizer.init(
        (eqx.filter(flow_map, eqx.is_array), eqx.filter(joint_loss.u_mlp, eqx.is_array))
    )

    flow_map = eqx.filter_shard(flow_map, replicate_sharding)
    ema_flow_map = eqx.filter_shard(ema_flow_map, replicate_sharding)
    joint_loss = eqx.filter_shard(joint_loss, replicate_sharding)
    opt_state = eqx.filter_shard(opt_state, replicate_sharding)

    start_step = 0
    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt is not None:
        flow_map, ema_flow_map, opt_state, u_mlp, start_step, key = load_checkpoint(
            latest_ckpt,
            flow_map,
            ema_flow_map,
            joint_loss.u_mlp,
            optimizer,
        )
        joint_loss = eqx.tree_at(lambda l: l.u_mlp, joint_loss, u_mlp)
        flow_map = eqx.filter_shard(flow_map, replicate_sharding)
        ema_flow_map = eqx.filter_shard(ema_flow_map, replicate_sharding)
        joint_loss = eqx.filter_shard(joint_loss, replicate_sharding)
        opt_state = eqx.filter_shard(opt_state, replicate_sharding)

    for step in range(start_step, n_train_steps):
        key, key_batch, key_step = jax.random.split(key, 3)
        idx = jax.random.randint(key_batch, (batch_size,), 0, dataset.shape[0])
        y0_batch = dataset[idx]
        y0_batch = jax.device_put(y0_batch, data_sharding)

        flow_map, ema_flow_map, opt_state, joint_loss, loss = train_step(
            flow_map,
            ema_flow_map,
            y0_batch,
            opt_state,
            optimizer,
            joint_loss,
            ema_decay,
            key_step,
        )

        if step % 1000 == 0:
            loss_val = loss.item()
            print(f"Step {step:>6d} | Loss: {loss_val:.4f}")
            wandb.log({"loss": loss_val}, step=step)

        if step > 0 and step % sample_every == 0:
            key, key_sample = jax.random.split(key)
            fig = make_sample_grid(
                ema_flow_map,
                t_eps,
                key_sample,
                data_shape,
                sample_step_sizes,
                data_mean,
                data_std,
            )
            wandb.log({"samples": wandb.Image(fig)}, step=step)
            plt.close(fig)

        if step > 0 and step % checkpoint_every == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, format_step(step)),
                flow_map,
                ema_flow_map,
                opt_state,
                joint_loss.u_mlp,
                step,
                key,
            )

    loss_val = loss.item()
    print(f"Step {n_train_steps:>6d} | Loss: {loss_val:.4f}")
    wandb.log({"loss": loss_val}, step=n_train_steps)

    save_checkpoint(
        os.path.join(checkpoint_dir, format_step(n_train_steps)),
        flow_map,
        ema_flow_map,
        opt_state,
        joint_loss.u_mlp,
        n_train_steps,
        key,
    )

    key, key_sample = jax.random.split(key)
    fig = make_sample_grid(
        ema_flow_map,
        t_eps,
        key_sample,
        data_shape,
        sample_step_sizes,
        data_mean,
        data_std,
    )
    fig.savefig("experiments/cifar10/cifar10_edm2_em_samples.png", dpi=150)
    print("Saved experiments/cifar10/cifar10_edm2_em_samples.png")
    wandb.log({"samples": wandb.Image(fig)}, step=n_train_steps)
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    main()
