import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sde import (
    SDEFlowMap,
    SDEStepModel,
    aggregate_to,
    coarse_flow_map_traj,
    fine_em_truth,
    sample_L,
)

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
    }
)


def shifted_legendre_antideriv(n: int, x):
    if n == 0:
        return x
    if n == 1:
        return x**2 - x
    if n == 2:
        return 2.0 * x**3 - 3.0 * x**2 + x
    if n == 3:
        return 5.0 * x**4 - 10.0 * x**3 + 6.0 * x**2 - x
    raise ValueError(f"unsupported n={n}")


def polynomial_brownian_reconstruction(
    L_coarse, n_coeffs: int, h_coarse: float, n_per_coarse: int = 200
):
    n_coarse = L_coarse.shape[0]

    xs = jnp.linspace(0.0, 1.0, n_per_coarse)
    basis_antideriv = jnp.stack(
        [shifted_legendre_antideriv(n, xs) for n in range(n_coeffs)]
    )
    weights = 2.0 * jnp.arange(n_coeffs) + 1.0
    weighted_L = L_coarse * weights[None, :]
    contributions = weighted_L @ basis_antideriv

    W_at_starts = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(L_coarse[:, 0])])
    W_poly = W_at_starts[:-1, None] + contributions

    interval_starts = jnp.arange(n_coarse) * h_coarse
    times = interval_starts[:, None] + xs[None, :] * h_coarse

    return np.asarray(times.reshape(-1)), np.asarray(W_poly.reshape(-1))


def make_polynomial_brownian_figure(
    seed: int = 4,
    n_fine: int = 1024,
    n_coarse: int = 16,
    color_truth: str = "0.35",
    alpha_truth: float = 0.7,
    alpha_poly: float = 1.0,
):
    N_values = (1, 2, 3, 4)
    poly_colors = ("tab:blue", "tab:orange", "tab:green", "tab:purple")
    n_coeffs_max = max(N_values)
    h_fine = 1.0 / n_fine
    h_coarse = 1.0 / n_coarse

    key = jax.random.PRNGKey(seed)
    _, _, key_L = jax.random.split(key, 3)
    keys_fine = jax.random.split(key_L, n_fine)
    L_fine = jax.vmap(lambda k: sample_L(n_coeffs_max, h_fine, k))(keys_fine)  # pyright: ignore

    L_coarse_full = aggregate_to(L_fine, n_coarse, n_coeffs_max)
    W_increments = L_fine[:, 0]
    W_path = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(W_increments)])
    fine_times = np.linspace(0.0, 1.0, n_fine + 1)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.4), sharex=True, sharey=True)

    for i, N in enumerate(N_values):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        color_poly = poly_colors[i]
        L_coarse_N = L_coarse_full[:, :N]
        poly_times, poly_W = polynomial_brownian_reconstruction(L_coarse_N, N, h_coarse)
        ax.plot(
            fine_times,
            np.asarray(W_path),
            color=color_truth,
            lw=0.7,
            alpha=alpha_truth,
            label=r"$\bm{W}_t$",
        )
        ax.plot(
            poly_times,
            poly_W,
            color=color_poly,
            lw=1.2,
            alpha=alpha_poly,
            label=rf"$\bm{{W}}_t^{{({N})}}$",
        )
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.legend(loc="upper right", borderaxespad=0.4)

        if col == 0:
            ax.set_ylabel("Brownian Path")
        if row == 1:
            ax.set_xlabel("Time")

    fig.tight_layout()

    os.makedirs("experiments/sde/figures", exist_ok=True)
    out_png = "experiments/sde/figures/polynomial_brownian.png"
    out_pdf = "experiments/sde/figures/polynomial_brownian.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_png} and {out_pdf}")


def main():
    n_coeffs = 4
    mlp_width = 64
    mlp_depth = 3

    n_fine = 1024
    coarse_step_counts = (2, 4, 8, 16, 32, 64)
    h_fine = 1.0 / n_fine

    color_truth = "0.35"
    color_flow = "tab:purple"
    alpha_truth = 0.7
    alpha_flow = 1.0

    seed = 4

    key = jax.random.PRNGKey(seed)
    key_init, key_x0, key_L = jax.random.split(key, 3)

    step_model = SDEStepModel(
        n_coeffs=n_coeffs, width=mlp_width, depth=mlp_depth, key=key_init
    )
    flow_map = SDEFlowMap(step_model=step_model)
    flow_map = eqx.tree_deserialise_leaves(
        f"experiments/sde/models/sde_N{n_coeffs}.eqx", flow_map
    )
    flow_map = eqx.nn.inference_mode(flow_map)

    x0 = jax.random.normal(key_x0, (1,))

    keys_fine = jax.random.split(key_L, n_fine)
    L_fine = jax.vmap(lambda k: sample_L(n_coeffs, h_fine, k))(keys_fine)  # pyright: ignore

    truth_traj = fine_em_truth(x0, L_fine[:, 0:1], h_fine)
    fine_times = np.linspace(0.0, 1.0, n_fine + 1)

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.4), sharex=True, sharey=True)

    truth_line = None
    flow_line = None
    for i, n_coarse in enumerate(coarse_step_counts):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        h_coarse = 1.0 / n_coarse
        L_coarse = aggregate_to(L_fine, n_coarse, n_coeffs)
        coarse_traj = coarse_flow_map_traj(flow_map, x0, L_coarse, h_coarse)
        coarse_times = np.linspace(0.0, 1.0, n_coarse + 1)

        (truth_line,) = ax.plot(
            fine_times,
            np.asarray(truth_traj[:, 0]),
            color=color_truth,
            lw=0.7,
            alpha=alpha_truth,
            label=r"Ground truth $\bm{X}_t$",
        )
        (flow_line,) = ax.plot(
            coarse_times,
            np.asarray(coarse_traj[:, 0]),
            color=color_flow,
            marker="o",
            ms=1.8,
            lw=1.0,
            alpha=alpha_flow,
            label=r"Flow Map $\bm{X}_t^\theta$",
        )
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_title(rf"$n={n_coarse}$")

        if col == 0:
            ax.set_ylabel("Solution")
        if row == 1:
            ax.set_xlabel("Time")

    fig.legend(
        handles=[truth_line, flow_line],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        borderaxespad=0.0,
    )
    fig.tight_layout()

    os.makedirs("experiments/sde/figures", exist_ok=True)
    out_png = f"experiments/sde/figures/main_figure_N{n_coeffs}.png"
    out_pdf = f"experiments/sde/figures/main_figure_N{n_coeffs}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
    make_polynomial_brownian_figure()
