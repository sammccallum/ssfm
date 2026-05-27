import argparse
import functools
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from ssfm.typing import Y

BETA_MIN = 0.1
BETA_MAX = 20.0


def _beta(t: Float[Array, ""]) -> Float[Array, ""]:
    return BETA_MIN + (BETA_MAX - BETA_MIN) * t


def b_drift(t: Float[Array, ""], x: Y) -> Y:
    return x - x**3


def sigma_diffusion(t: Float[Array, ""]) -> Float[Array, ""]:
    return jnp.sqrt(_beta(t))


C_MATRIX = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0],
        [0.0, -3.0 / 4.0, 1.0 / 4.0, 0.0],
        [1.0 / 8.0, 3.0 / 8.0, -5.0 / 8.0, 1.0 / 8.0],
    ]
)
SIGN_MATRIX = jnp.array([[(-1.0) ** (n + m) for m in range(4)] for n in range(4)])


def sample_L(
    n_coeffs: int, h: Float[Array, ""], key: PRNGKeyArray
) -> Float[Array, " n"]:
    indices = jnp.arange(n_coeffs)
    stds = jnp.sqrt(h / (2.0 * indices + 1.0))
    z = jax.random.normal(key, (n_coeffs,))
    return stds * z


def chen_combine(
    L_left: Float[Array, " n"], L_right: Float[Array, " n"], n_coeffs: int
) -> Float[Array, " n"]:
    c = C_MATRIX[:n_coeffs, :n_coeffs]
    sign = SIGN_MATRIX[:n_coeffs, :n_coeffs]
    return c @ L_left + (c * sign) @ L_right


def aggregate_pairwise(
    L: Float[Array, "n_intervals n_coeffs"], n_coeffs: int
) -> Float[Array, "n_intervals_half n_coeffs"]:
    L_left = L[0::2]
    L_right = L[1::2]
    return jax.vmap(lambda l, r: chen_combine(l, r, n_coeffs))(L_left, L_right)


def aggregate_to(
    L_fine: Float[Array, "n_fine n_coeffs"], n_target: int, n_coeffs: int
) -> Float[Array, "n_target n_coeffs"]:
    L = L_fine
    while L.shape[0] > n_target:
        L = aggregate_pairwise(L, n_coeffs)
    return L


class SDEStepModel(eqx.Module):
    drift_mlp: eqx.nn.MLP
    diffusion_mlp: eqx.nn.MLP
    n_coeffs: int = eqx.field(static=True)

    def __init__(self, n_coeffs: int, width: int, depth: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.n_coeffs = n_coeffs
        self.drift_mlp = eqx.nn.MLP(
            in_size=3 + n_coeffs,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=k1,
        )
        self.diffusion_mlp = eqx.nn.MLP(
            in_size=2 + n_coeffs,
            out_size=1,
            width_size=width,
            depth=depth,
            activation=jax.nn.silu,
            key=k2,
        )

    def drift(
        self, y: Y, s: Float[Array, ""], t: Float[Array, ""], L: Float[Array, " n"]
    ) -> Y:
        x = jnp.concatenate([y, s[None], t[None], L])
        return self.drift_mlp(x)

    def diffusion(
        self, s: Float[Array, ""], t: Float[Array, ""], L: Float[Array, " n"]
    ) -> Y:
        x = jnp.concatenate([s[None], t[None], L])
        return self.diffusion_mlp(x)


class SDEFlowMap(eqx.Module):
    step_model: SDEStepModel

    def __call__(
        self, y: Y, s: Float[Array, ""], t: Float[Array, ""], L: Float[Array, " n"]
    ) -> Y:
        h = t - s
        W = L[0:1]
        return (
            y
            + h * self.step_model.drift(y, s, t, L)
            + W * self.step_model.diffusion(s, t, L)
        )


def fine_em_to_xs(x0: Y, s: Float[Array, ""], key: PRNGKeyArray, n_sub: int) -> Y:
    h_sub = s / n_sub
    keys = jax.random.split(key, n_sub)

    def step(carry, key_step):
        x, t = carry
        z = jax.random.normal(key_step, x.shape)
        x_new = x + b_drift(t, x) * h_sub + sigma_diffusion(t) * jnp.sqrt(h_sub) * z
        return (x_new, t + h_sub), None

    (xs, _), _ = jax.lax.scan(step, (x0, jnp.array(0.0)), keys)
    return xs


class SDEScoreLoss(eqx.Module):
    dt: float
    n_coeffs: int = eqx.field(static=True)
    n_sub: int = eqx.field(static=True)
    t_eps: float = 1e-5

    def _single(
        self,
        flow_map: SDEFlowMap,
        x0: Y,
        s: Float[Array, ""],
        h: Float[Array, ""],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        em_key, l_key = jax.random.split(key)
        xs = fine_em_to_xs(x0, s, em_key, self.n_sub)

        t = s + h
        L = sample_L(self.n_coeffs, h, l_key)
        W = L[0:1]

        target = xs + b_drift(s, xs) * h + sigma_diffusion(s) * W
        pred = flow_map(xs, s, t, L)

        return jnp.sum((pred - jax.lax.stop_gradient(target)) ** 2) / h

    def value(
        self,
        flow_map: SDEFlowMap,
        x0_batch: Float[Array, "batch 1"],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        batch_size = x0_batch.shape[0]
        key_h, key_s, key_step = jax.random.split(key, 3)

        h_vals = jax.random.uniform(
            key_h, (batch_size,), minval=self.dt / 2.0, maxval=self.dt
        )
        s_vals = jax.random.uniform(
            key_s, (batch_size,), minval=self.t_eps, maxval=1.0 - h_vals
        )
        keys = jax.random.split(key_step, batch_size)

        losses = eqx.filter_vmap(
            lambda x0, s, h, k: self._single(flow_map, x0, s, h, k)
        )(x0_batch, s_vals, h_vals, keys)
        return jnp.mean(losses)


class SDEDistillationLoss(eqx.Module):
    dt: float
    h_max: float
    n_coeffs: int = eqx.field(static=True)
    n_sub: int = eqx.field(static=True)
    t_eps: float = 1e-5

    def _single(
        self,
        flow_map: SDEFlowMap,
        ema_flow_map: SDEFlowMap,
        x0: Y,
        s: Float[Array, ""],
        h: Float[Array, ""],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        em_key, l_left_key, l_right_key = jax.random.split(key, 3)
        ema_flow_map = eqx.nn.inference_mode(ema_flow_map)

        xs = fine_em_to_xs(x0, s, em_key, self.n_sub)

        t = s + h
        mid = s + h / 2.0

        L_left = sample_L(self.n_coeffs, h / 2.0, l_left_key)
        L_right = sample_L(self.n_coeffs, h / 2.0, l_right_key)
        L_full = chen_combine(L_left, L_right, self.n_coeffs)

        x_mid = ema_flow_map(xs, s, mid, L_left)
        target = ema_flow_map(x_mid, mid, t, L_right)
        target = jax.lax.stop_gradient(target)

        pred = flow_map(xs, s, t, L_full)
        return jnp.sum((pred - target) ** 2) / h

    def value(
        self,
        flow_map: SDEFlowMap,
        ema_flow_map: SDEFlowMap,
        x0_batch: Float[Array, "batch 1"],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        batch_size = x0_batch.shape[0]
        key_h, key_s, key_step = jax.random.split(key, 3)

        h_vals = jax.random.uniform(
            key_h, (batch_size,), minval=self.dt, maxval=self.h_max
        )
        s_max = jnp.maximum(1.0 - h_vals, self.t_eps + 1e-6)
        s_vals = jax.random.uniform(
            key_s, (batch_size,), minval=self.t_eps, maxval=s_max
        )
        keys = jax.random.split(key_step, batch_size)

        losses = eqx.filter_vmap(
            lambda x0, s, h, k: self._single(flow_map, ema_flow_map, x0, s, h, k)
        )(x0_batch, s_vals, h_vals, keys)
        return jnp.mean(losses)


class SDEJointLoss(eqx.Module):
    score_loss: SDEScoreLoss
    distill_loss: SDEDistillationLoss
    eta: float = 0.75

    def value(
        self,
        flow_map: SDEFlowMap,
        ema_flow_map: SDEFlowMap,
        x0_batch: Float[Array, "batch 1"],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        key_score, key_distill = jax.random.split(key)

        n_score = int(x0_batch.shape[0] * self.eta)
        score_batch = x0_batch[:n_score]
        distill_batch = x0_batch[n_score:]

        l_score = self.score_loss.value(flow_map, score_batch, key_score)
        l_distill = self.distill_loss.value(
            flow_map, ema_flow_map, distill_batch, key_distill
        )

        return self.eta * l_score + (1.0 - self.eta) * l_distill

    def value_and_grad(
        self,
        flow_map: SDEFlowMap,
        ema_flow_map: SDEFlowMap,
        x0_batch: Float[Array, "batch 1"],
        key: PRNGKeyArray,
    ):
        @eqx.filter_value_and_grad
        def _loss(fm):
            return self.value(fm, ema_flow_map, x0_batch, key)

        return _loss(flow_map)


def ema_update(model: SDEFlowMap, ema_model: SDEFlowMap, decay: float) -> SDEFlowMap:
    params, static = eqx.partition(model, eqx.is_array)
    ema_params, _ = eqx.partition(ema_model, eqx.is_array)
    new_ema_params = jax.tree.map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params
    )
    return eqx.combine(new_ema_params, static)


@eqx.filter_jit
def train_step(
    flow_map: SDEFlowMap,
    ema_flow_map: SDEFlowMap,
    x0_batch: Float[Array, "batch 1"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    joint_loss: SDEJointLoss,
    ema_decay: float,
    key: PRNGKeyArray,
):
    loss, grads = joint_loss.value_and_grad(flow_map, ema_flow_map, x0_batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, flow_map)
    flow_map = eqx.apply_updates(flow_map, updates)
    ema_flow_map = ema_update(flow_map, ema_flow_map, ema_decay)
    return flow_map, ema_flow_map, opt_state, loss


def fine_em_truth(
    x0: Y, L_fine_W: Float[Array, "n_fine 1"], h_fine: float
) -> Float[Array, "n_fine_plus_1 1"]:
    def step(carry, w):
        x, t = carry
        x_new = x + b_drift(t, x) * h_fine + sigma_diffusion(t) * w
        return (x_new, t + h_fine), x_new

    (_, _), traj = jax.lax.scan(step, (x0, jnp.array(0.0)), L_fine_W)
    return jnp.concatenate([x0[None], traj], axis=0)


def coarse_flow_map_traj(
    flow_map: SDEFlowMap,
    x0: Y,
    L_coarse: Float[Array, "n_coarse n_coeffs"],
    h_coarse: float,
) -> Float[Array, "n_coarse_plus_1 1"]:
    def step(carry, L):
        x, t = carry
        x_new = flow_map(x, t, t + h_coarse, L)
        return (x_new, t + h_coarse), x_new

    (_, _), traj = jax.lax.scan(step, (x0, jnp.array(0.0)), L_coarse)
    return jnp.concatenate([x0[None], traj], axis=0)


@functools.partial(jax.jit, static_argnames=("n_coeffs", "n_traj", "n_fine"))
def sample_eval_paths(
    n_coeffs: int, n_traj: int, n_fine: int, h_fine: float, key: PRNGKeyArray
):
    key_x0, key_L = jax.random.split(key)
    x0_batch = jax.random.normal(key_x0, (n_traj, 1))

    keys_2d = jax.random.split(key_L, n_traj * n_fine).reshape(n_traj, n_fine, 2)
    L_fine_batch = jax.vmap(jax.vmap(lambda k: sample_L(n_coeffs, h_fine, k)))(keys_2d)

    truth_trajs = jax.vmap(lambda x0, L: fine_em_truth(x0, L[:, 0:1], h_fine))(
        x0_batch, L_fine_batch
    )

    return x0_batch, L_fine_batch, truth_trajs


@functools.partial(eqx.filter_jit, donate="none")
def coarse_eval_one_size(
    flow_map: SDEFlowMap,
    x0_batch: Float[Array, "n_traj 1"],
    L_fine_batch: Float[Array, "n_traj n_fine n_coeffs"],
    truth_trajs: Float[Array, "n_traj n_fine_plus_1 1"],
    n_coarse: int,
    n_coeffs: int,
    h_coarse: float,
    stride: int,
):
    L_coarse = jax.vmap(lambda L: aggregate_to(L, n_coarse, n_coeffs))(L_fine_batch)
    coarse_trajs = jax.vmap(
        lambda x0, L: coarse_flow_map_traj(flow_map, x0, L, h_coarse)
    )(x0_batch, L_coarse)

    truth_at_coarse = truth_trajs[:, ::stride, :]
    mse_per_traj = jnp.mean((coarse_trajs - truth_at_coarse) ** 2, axis=(1, 2))
    return mse_per_traj, coarse_trajs


def evaluate(
    flow_map: SDEFlowMap,
    n_coeffs: int,
    n_traj: int,
    n_fine: int,
    eval_step_sizes: list[float],
    key: PRNGKeyArray,
):
    h_fine = 1.0 / n_fine
    x0_batch, L_fine_batch, truth_trajs = sample_eval_paths(
        n_coeffs, n_traj, n_fine, h_fine, key
    )

    results: dict = {}
    for h_coarse in eval_step_sizes:
        n_coarse = int(round(1.0 / h_coarse))
        stride = n_fine // n_coarse
        mse_per_traj, coarse_trajs = coarse_eval_one_size(
            flow_map,
            x0_batch,
            L_fine_batch,
            truth_trajs,
            n_coarse,
            n_coeffs,
            h_coarse,
            stride,
        )
        results[h_coarse] = {
            "mse_mean": float(jnp.mean(mse_per_traj)),
            "mse_std": float(jnp.std(mse_per_traj)),
            "trajs_coarse": np.asarray(coarse_trajs),
        }

    return results, np.asarray(truth_trajs)


def plot_trajectory_overlay(
    truth_trajs: np.ndarray,
    results: dict,
    eval_step_sizes: list[float],
    n_show: int = 4,
) -> plt.Figure:
    n_fine_plus_1 = truth_trajs.shape[1]
    fine_times = np.linspace(0.0, 1.0, n_fine_plus_1)

    n_steps = len(eval_step_sizes)
    fig, axes = plt.subplots(
        n_show,
        n_steps,
        figsize=(2.5 * n_steps, 2.0 * n_show),
        sharex=True,
        sharey="row",
    )
    if n_show == 1:
        axes = axes[None, :]
    if n_steps == 1:
        axes = axes[:, None]

    for i in range(n_show):
        row_min = float(truth_trajs[i, :, 0].min())
        row_max = float(truth_trajs[i, :, 0].max())
        for h_coarse in eval_step_sizes:
            traj = results[h_coarse]["trajs_coarse"][i, :, 0]
            row_min = min(row_min, float(traj.min()))
            row_max = max(row_max, float(traj.max()))
        pad = 0.1 * max(row_max - row_min, 1e-3)
        ylim = (row_min - pad, row_max + pad)

        for j, h_coarse in enumerate(eval_step_sizes):
            n_coarse = int(round(1.0 / h_coarse))
            coarse_times = np.linspace(0.0, 1.0, n_coarse + 1)
            ax = axes[i, j]
            ax.plot(fine_times, truth_trajs[i, :, 0], color="black", alpha=0.6, lw=1.0)
            ax.plot(
                coarse_times,
                results[h_coarse]["trajs_coarse"][i, :, 0],
                color="red",
                marker=".",
                lw=1.0,
                ms=3,
            )
            ax.set_ylim(ylim)
            if i == 0:
                ax.set_title(f"h={h_coarse:.4f}", fontsize=9)
    fig.suptitle("Truth (black) vs flow map (red)", y=1.0)
    fig.tight_layout()
    return fig


def run_sanity_checks() -> None:
    print("Running sanity checks ...")
    key = jax.random.PRNGKey(123)
    h = 0.7
    n_samples = 200_000

    keys = jax.random.split(key, n_samples)
    L_samples = jax.vmap(lambda k: sample_L(4, h, k))(keys)
    empirical_var = jnp.var(L_samples, axis=0)
    expected_var = h / (2.0 * jnp.arange(4) + 1.0)
    print(f"  (i)  empirical Var(L^(0..3)) = {empirical_var.tolist()}")
    print(f"       expected               = {expected_var.tolist()}")
    rel_err = jnp.max(jnp.abs(empirical_var - expected_var) / expected_var)
    assert rel_err < 0.05, f"variance check failed; rel_err = {rel_err}"

    key1, key2 = jax.random.split(jax.random.PRNGKey(7))
    keys_left = jax.random.split(key1, n_samples)
    keys_right = jax.random.split(key2, n_samples)
    L_left = jax.vmap(lambda k: sample_L(4, h / 2.0, k))(keys_left)
    L_right = jax.vmap(lambda k: sample_L(4, h / 2.0, k))(keys_right)
    L_combined = jax.vmap(lambda l, r: chen_combine(l, r, 4))(L_left, L_right)
    var_combined = jnp.var(L_combined, axis=0)
    print(f"  (ii) Chen-combined Var      = {var_combined.tolist()}")
    rel_err = jnp.max(jnp.abs(var_combined - expected_var) / expected_var)
    assert rel_err < 0.05, f"Chen-combine variance check failed; rel_err = {rel_err}"

    n_fine = 1024
    h_fine = 1.0 / n_fine
    key_a = jax.random.PRNGKey(42)
    keys_fine = jax.random.split(key_a, n_samples * n_fine).reshape(
        n_samples, n_fine, 2
    )
    L_fine = jax.vmap(jax.vmap(lambda k: sample_L(4, h_fine, k)))(keys_fine)
    L_full = jax.vmap(lambda Lf: aggregate_to(Lf, 1, 4))(L_fine)[:, 0, :]
    var_full = jnp.var(L_full, axis=0)
    expected_full = 1.0 / (2.0 * jnp.arange(4) + 1.0)
    print(f"  (iii) aggregated Var (h=1)  = {var_full.tolist()}")
    print(f"        expected              = {expected_full.tolist()}")
    rel_err = jnp.max(jnp.abs(var_full - expected_full) / expected_full)
    assert rel_err < 0.05, f"aggregation variance check failed; rel_err = {rel_err}"

    print("All sanity checks passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-coeffs", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true", help="short training run")
    parser.add_argument(
        "--sanity", action="store_true", help="run sanity checks and exit"
    )
    args = parser.parse_args()

    if args.sanity:
        run_sanity_checks()
        return

    n_coeffs = args.n_coeffs
    smoke = args.smoke

    data_dim = 1
    batch_size = 4096
    n_train_steps = 500 if smoke else 100_000
    lr = 1e-3
    eta = 0.75
    ema_decay = 0.999
    dt = 1e-3
    h_max = 1.0
    n_sub = 32
    mlp_width = 64
    mlp_depth = 3
    log_every = 100 if smoke else 500
    sample_every = 100 if smoke else 5_000

    n_fine = 1024
    eval_step_sizes = [1.0, 1 / 2, 1 / 4, 1 / 8, 1 / 16]
    n_eval_traj = 32 if smoke else 256

    config = {
        "n_coeffs": n_coeffs,
        "data_dim": data_dim,
        "batch_size": batch_size,
        "n_train_steps": n_train_steps,
        "lr": lr,
        "eta": eta,
        "ema_decay": ema_decay,
        "dt": dt,
        "h_max": h_max,
        "n_sub": n_sub,
        "mlp_width": mlp_width,
        "mlp_depth": mlp_depth,
        "n_fine": n_fine,
        "eval_step_sizes": eval_step_sizes,
        "n_eval_traj": n_eval_traj,
        "beta_min": BETA_MIN,
        "beta_max": BETA_MAX,
    }

    key = jax.random.PRNGKey(0)
    key, key_model = jax.random.split(key)

    step_model = SDEStepModel(
        n_coeffs=n_coeffs, width=mlp_width, depth=mlp_depth, key=key_model
    )
    flow_map = SDEFlowMap(step_model=step_model)
    ema_flow_map = flow_map

    score_loss = SDEScoreLoss(dt=dt, n_coeffs=n_coeffs, n_sub=n_sub)
    distill_loss = SDEDistillationLoss(
        dt=dt, h_max=h_max, n_coeffs=n_coeffs, n_sub=n_sub
    )
    joint_loss = SDEJointLoss(score_loss=score_loss, distill_loss=distill_loss, eta=eta)

    lr_min = 1e-5
    schedule = optax.cosine_decay_schedule(
        init_value=lr, decay_steps=n_train_steps, alpha=lr_min / lr
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(eqx.filter(flow_map, eqx.is_array))

    os.makedirs("experiments/sde/models", exist_ok=True)
    os.makedirs("experiments/sde/figures", exist_ok=True)
    os.makedirs("experiments/sde/logs", exist_ok=True)

    with open(f"experiments/sde/logs/config_N{n_coeffs}.json", "w") as f:
        json.dump(config, f, indent=2)

    loss_log: list[tuple[int, float]] = []
    mse_log: list[dict] = []

    for step in range(n_train_steps):
        key, key_x0, key_loss = jax.random.split(key, 3)
        x0_batch = jax.random.normal(key_x0, (batch_size, data_dim))

        flow_map, ema_flow_map, opt_state, loss = train_step(
            flow_map,
            ema_flow_map,
            x0_batch,
            opt_state,
            optimizer,
            joint_loss,
            ema_decay,
            key_loss,
        )

        if step % log_every == 0:
            loss_val = float(loss)
            print(f"Step {step:>6d} | Loss: {loss_val:.6f}")
            loss_log.append((step, loss_val))

        if step > 0 and step % sample_every == 0:
            key, key_eval = jax.random.split(key)
            results, truth_trajs = evaluate(
                ema_flow_map, n_coeffs, n_eval_traj, n_fine, eval_step_sizes, key_eval
            )

            fig = plot_trajectory_overlay(truth_trajs, results, eval_step_sizes)
            fig.savefig(
                f"experiments/sde/figures/trajectories_N{n_coeffs}.png", dpi=150
            )
            plt.close(fig)

            mse_entry = {
                "step": step,
                "mse": {h: results[h]["mse_mean"] for h in eval_step_sizes},
            }
            mse_log.append(mse_entry)
            mse_summary = ", ".join(
                f"h={h:.4f}:{results[h]['mse_mean']:.3e}" for h in eval_step_sizes
            )
            print(f"  eval @ step {step}: {mse_summary}")

    print(f"Step {n_train_steps:>6d} | Final loss: {float(loss):.6f}")

    eqx.tree_serialise_leaves(
        f"experiments/sde/models/sde_N{n_coeffs}.eqx", ema_flow_map
    )
    print(f"Saved model to experiments/sde/models/sde_N{n_coeffs}.eqx")

    key, key_eval = jax.random.split(key)
    results, truth_trajs = evaluate(
        ema_flow_map, n_coeffs, n_eval_traj, n_fine, eval_step_sizes, key_eval
    )

    fig = plot_trajectory_overlay(truth_trajs, results, eval_step_sizes)
    fig.savefig(f"experiments/sde/figures/trajectories_N{n_coeffs}.png", dpi=150)
    plt.close(fig)

    final_mse = {h: results[h]["mse_mean"] for h in eval_step_sizes}
    print("Final MSE by step size:")
    for h in eval_step_sizes:
        print(
            f"  h={h:.6f}: {results[h]['mse_mean']:.4e} (std {results[h]['mse_std']:.4e})"
        )

    fine_eval_step_sizes = [1.0 / 32, 1.0 / 64, 1.0 / 128, 1.0 / 256, 1.0 / 512]
    fine_n_fine = 4096
    key, key_fine = jax.random.split(key)
    fine_results, _ = evaluate(
        ema_flow_map,
        n_coeffs,
        n_eval_traj,
        fine_n_fine,
        fine_eval_step_sizes,
        key_fine,
    )

    fine_h_arr = np.array(sorted(fine_eval_step_sizes))
    fine_rmse_arr = np.array([np.sqrt(fine_results[h]["mse_mean"]) for h in fine_h_arr])
    fine_order, _ = np.polyfit(np.log(fine_h_arr), np.log(fine_rmse_arr), 1)

    print("Fine-regime MSE (32-256 steps):")
    for h in fine_eval_step_sizes:
        n_steps = int(round(1.0 / h))
        print(
            f"  {n_steps:>3d} steps (h={h:.5f}): MSE={fine_results[h]['mse_mean']:.4e} "
            f"(std {fine_results[h]['mse_std']:.4e})"
        )
    print(f"Fine-regime fitted strong order γ ≈ {fine_order:.3f}")

    final_fine_mse = {h: fine_results[h]["mse_mean"] for h in fine_eval_step_sizes}

    with open(f"experiments/sde/logs/loss_N{n_coeffs}.json", "w") as f:
        json.dump(loss_log, f, indent=2)
    with open(f"experiments/sde/logs/mse_N{n_coeffs}.json", "w") as f:
        json.dump(
            {
                "during_training": mse_log,
                "final": final_mse,
                "final_fine": final_fine_mse,
                "fine_strong_order": float(fine_order),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
