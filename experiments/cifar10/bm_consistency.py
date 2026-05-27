import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from eval_fid import build_model
from main import cifar10

from ssfm.flow_map import EulerMaruyamaFlowMap

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 20,
        "axes.titlesize": 26,
    }
)


def integrate_with_shared_bm(
    flow_map: EulerMaruyamaFlowMap,
    y_init: jnp.ndarray,
    vbt: diffrax.VirtualBrownianTree,
    s: float,
    t_eps: float,
    n_steps: int,
) -> jnp.ndarray:
    flow_map = eqx.nn.inference_mode(flow_map)
    time_grid = jnp.linspace(s, t_eps, n_steps + 1)

    def scan_fn(y, i):
        s_i = jnp.clip(time_grid[i], t_eps, s)
        t_i = jnp.clip(time_grid[i + 1], t_eps, s)
        levy = vbt.evaluate(s_i, t_i, use_levy=True)
        y = flow_map(y, s_i, t_i, levy.W, levy.H, levy.K)  # pyright: ignore
        return y, None

    y_final, _ = jax.lax.scan(scan_fn, y_init, jnp.arange(n_steps))
    return y_final


def to_image(arr: jnp.ndarray, data_mean: jnp.ndarray, data_std: jnp.ndarray):
    img = arr * data_std + data_mean
    img = jnp.clip(img, 0.0, 1.0)
    return np.array(img.transpose(1, 2, 0))


def main(
    checkpoint_path: str = "experiments/cifar10/checkpoints/350k",
    step_counts: tuple[int, ...] = (2, 4, 8, 16, 32),
    t_eps: float = 1e-5,
    s: float = 1.0,
    seed: int = 8,
    save_path: str = "experiments/cifar10/bm_consistency.pdf",
):
    in_channels = 3
    resolution = 32
    data_shape = (in_channels, resolution, resolution)

    _, data_mean, data_std = cifar10()
    data_mean = data_mean.reshape(data_shape)
    data_std = data_std.reshape(data_shape)

    flow_map = build_model(
        jax.random.PRNGKey(0), in_channels=in_channels, resolution=resolution
    )
    ema_path = os.path.join(checkpoint_path, "ema_flow_map.eqx")
    flow_map = eqx.tree_deserialise_leaves(ema_path, flow_map)
    print(f"Loaded EMA model from {ema_path}")

    smallest_step = (s - t_eps) / max(step_counts)
    bm_tol = smallest_step / 16
    n_cols = len(step_counts)
    n_fixed = max(step_counts)

    key = jax.random.PRNGKey(seed)
    key_init, key_top_bm, key_bottom_bms = jax.random.split(key, 3)
    y_init = jax.random.normal(key_init, data_shape)

    vbt_top = diffrax.VirtualBrownianTree(
        t0=t_eps,
        t1=s,
        tol=bm_tol,
        shape=data_shape,
        key=key_top_bm,
        levy_area=diffrax.SpaceTimeTimeLevyArea,
    )
    top_samples = jnp.stack(
        [
            integrate_with_shared_bm(flow_map, y_init, vbt_top, s, t_eps, n)
            for n in step_counts
        ]
    )

    bm_keys = jax.random.split(key_bottom_bms, n_cols)

    def integrate_with_new_bm(bm_key):
        vbt = diffrax.VirtualBrownianTree(
            t0=t_eps,
            t1=s,
            tol=bm_tol,
            shape=data_shape,
            key=bm_key,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )
        return integrate_with_shared_bm(flow_map, y_init, vbt, s, t_eps, n_fixed)

    bottom_samples = jax.vmap(integrate_with_new_bm)(bm_keys)

    fig, axes = plt.subplots(2, n_cols, figsize=(2 * n_cols, 2 * 2))
    for c, n in enumerate(step_counts):
        ax = axes[0, c]
        ax.imshow(to_image(top_samples[c], data_mean, data_std))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rf"$n={n}$")
    for c in range(n_cols):
        ax = axes[1, c]
        ax.imshow(to_image(bottom_samples[c], data_mean, data_std))
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_ylabel(r"Same $\mathbf{W}_t$")
    axes[1, 0].set_ylabel(r"Different $\mathbf{W}_t$")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    png_path = os.path.splitext(save_path)[0] + ".png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved {save_path} and {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
