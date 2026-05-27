import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
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

LOG_DIR = "experiments/sde/logs"
FIG_DIR = "experiments/sde/figures"

N_VALUES = [1, 2, 3, 4]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
MARKERS = ["o", "s", "^", "D"]
ALPHA = 0.85


def load_final_mse(n_coeffs: int) -> dict[float, float]:
    with open(f"{LOG_DIR}/mse_N{n_coeffs}.json") as f:
        data = json.load(f)
    return {float(h): float(v) for h, v in data["final"].items()}


def main():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))

    for N, color, marker in zip(N_VALUES, COLORS, MARKERS):
        mse = load_final_mse(N)
        h_arr = np.array(sorted(mse.keys(), reverse=True))
        steps = np.round(1.0 / h_arr).astype(int)
        mse_arr = np.array([mse[h] for h in h_arr])

        ax.plot(
            steps,
            mse_arr,
            color=color,
            alpha=ALPHA,
            marker=marker,
            ms=3.0,
            lw=1.2,
            label=rf"$\bm{{W}}_t^{{({N})}}$",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels([1, 2, 4, 8, 16])  # pyright: ignore
    ax.set_xlabel("Steps")
    ax.set_ylabel(r"Strong Error, $\mathbb{E}\|\bm{X}_t - \bm{X}_t^\theta\|^2$")
    ax.legend(loc="best")

    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = f"{FIG_DIR}/coarse_errors.png"
    out_pdf = f"{FIG_DIR}/coarse_errors.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
