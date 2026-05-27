"""Project Chignolin 10k samples to reference tICA and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import ot
from matplotlib.colors import LogNorm

from sampling import _safe_name, _save_figure
from utils import compute_tica_pmf_mjs

try:
    import wandb
except ImportError:  # pragma: no cover - optional logging dependency.
    wandb = None


DEFAULT_DATA_PATH = Path("/mnt/labs/data/tong/many-peptides-md/trajectories_subsampled/test/10AA/GYDPETGTWG_subsampled.npz")
DEFAULT_RESULTS_DIR = Path("experiments/chignolin/runs/results")
DEFAULT_STEPS = (100, 64, 32, 16, 8, 4, 2, 1)
DEFAULT_MODELS = (
    ("xpred-diffusion", "Diffusion"),
    ("strong-sfm-eta0.75", "SSFM"),
)


def _load_reference(path: Path, n_frames: int | None) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
        sequence = str(np.asarray(data["sequence"]).item())
    if n_frames is not None:
        positions = positions[: int(n_frames)]
    return positions, sequence


def _load_samples(path: Path, n_frames: int | None) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        samples = np.asarray(data["positions"], dtype=np.float64)
    if n_frames is not None:
        samples = samples[: int(n_frames)]
    return samples


def _exact_tica_w2(reference_tica: np.ndarray, sample_tica: np.ndarray) -> float:
    reference_tica = np.asarray(reference_tica, dtype=np.float64)
    sample_tica = np.asarray(sample_tica, dtype=np.float64)
    reference_tica = reference_tica[np.all(np.isfinite(reference_tica[:, :2]), axis=1), :2]
    sample_tica = sample_tica[np.all(np.isfinite(sample_tica[:, :2]), axis=1), :2]
    if reference_tica.shape[0] == 0 or sample_tica.shape[0] == 0:
        return float("nan")
    reference_weights = np.full(reference_tica.shape[0], 1.0 / reference_tica.shape[0], dtype=np.float64)
    sample_weights = np.full(sample_tica.shape[0], 1.0 / sample_tica.shape[0], dtype=np.float64)
    cost = ot.dist(reference_tica, sample_tica, metric="sqeuclidean")
    return math.sqrt(float(ot.emd2(reference_weights, sample_weights, cost, numItermax=int(1e7))))


def _hist2d(ax, tica: np.ndarray, limits: tuple[tuple[float, float], tuple[float, float]], title: str, bins: int) -> None:
    finite = np.all(np.isfinite(tica[:, :2]), axis=1)
    hist, _, _ = np.histogram2d(
        tica[finite, 0],
        tica[finite, 1],
        bins=int(bins),
        range=[list(limits[0]), list(limits[1])],
    )
    positive = hist[hist > 0.0]
    norm = None if positive.size == 0 else LogNorm(vmin=max(float(positive.min()), 1.0), vmax=float(positive.max()))
    ax.hist2d(
        tica[finite, 0],
        tica[finite, 1],
        bins=int(bins),
        range=[list(limits[0]), list(limits[1])],
        norm=norm,
        rasterized=True,
    )
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("auto")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("TIC 0", fontsize=8)
    ax.set_ylabel("TIC 1", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_tica_comparison(
    *,
    reference_tica: np.ndarray,
    models: list[dict[str, Any]],
    limits: tuple[tuple[float, float], tuple[float, float]],
    out_path: Path,
    bins: int,
) -> dict[str, Any]:
    column_steps = [steps for steps, _ in models[0]["tica_sets"]]
    columns = ["data"] + [f"{steps} steps" for steps in column_steps]
    fig, axes = plt.subplots(len(models), len(columns), figsize=(3.0 * len(columns), 2.8 * len(models)), squeeze=False)
    for row_idx, model in enumerate(models):
        sample_by_step = {steps: tica for steps, tica in model["tica_sets"]}
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx, col_idx]
            tica = reference_tica if col_idx == 0 else sample_by_step[column_steps[col_idx - 1]]
            _hist2d(ax, tica, limits, column if row_idx == 0 else "", bins)
            if col_idx == 0:
                ax.set_ylabel(model["display_name"], fontsize=9)
    fig.tight_layout()
    paths = _save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return paths


def plot_tica_w2_by_steps(*, models: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for model in models:
        metrics = model["tica_metrics"]
        steps = sorted(int(step) for step in metrics)
        values = np.asarray([float(metrics[step]["w2"]) for step in steps], dtype=np.float64)
        finite = np.isfinite(values)
        if finite.any():
            ax.plot(np.asarray(steps)[finite], values[finite], marker="o", linewidth=1.8, label=model["display_name"])
        if (~finite).any():
            ax.scatter(np.asarray(steps)[~finite], np.zeros((~finite).sum()), marker="x", color="tab:red")
    all_steps = sorted({int(step) for model in models for step in model["tica_metrics"]})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_steps)
    ax.set_xticklabels([str(step) for step in all_steps])
    ax.set_xlabel("sample steps")
    ax.set_ylabel("tICA W2")
    ax.set_title("Chignolin tICA Wasserstein")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = _save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return paths


def write_tica_w2_latex_table(*, models: list[dict[str, Any]], out_path: Path, precision: int = 3) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_steps = sorted({int(step) for model in models for step in model["tica_metrics"]})
    columns = "ll" + "r" * len(sample_steps)
    nfe_start = 3
    nfe_end = nfe_start + len(sample_steps) - 1
    best_by_step = {
        step: min(float(model["tica_metrics"][step]["w2"]) for model in models if np.isfinite(float(model["tica_metrics"][step]["w2"])))
        for step in sample_steps
    }
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{tICA W$_2$ ($\downarrow$) on Chignolin across step counts.}",
        r"  \label{tab:chignolin_tica_w2}",
        r"  \begin{tabular}{" + columns + "}",
        r"  \toprule",
        rf"   & & \multicolumn{{{len(sample_steps)}}}{{c}}{{NFE}} \\",
        rf"  \cmidrule(lr){{{nfe_start}-{nfe_end}}}",
        "   Metric & Method & " + " & ".join(str(step) for step in sample_steps) + r" \\",
        r"  \midrule",
    ]
    for idx, model in enumerate(models):
        values = []
        for step in sample_steps:
            value = float(model["tica_metrics"][step]["w2"])
            if not np.isfinite(value):
                values.append(r"\mathrm{nan}")
            elif np.isclose(value, best_by_step[step]):
                values.append(rf"\textbf{{{value:.{precision}f}}}")
            else:
                values.append(f"{value:.{precision}f}")
        metric = r"\multirow{2}{*}{tICA W$_2$}" if idx == 0 else ""
        method = model["display_name"]
        if method == "SSFM":
            method = r"\textbf{SSFM}"
        lines.append("   " + metric + " & " + method + " & " + " & ".join(values) + r" \\")
    lines.extend([r"  \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    table = "\n".join(lines)
    out_path.write_text(table, encoding="utf-8")
    return str(out_path)


def run(args: argparse.Namespace) -> None:
    results_dir = args.results_dir
    out_dir = results_dir / "tica"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = tuple(int(step) for step in args.sample_steps)
    reference_coords, sequence = _load_reference(args.data_path, args.n_frames)

    models: list[dict[str, Any]] = []
    for model_name, display_name in DEFAULT_MODELS:
        sample_sets = []
        for step in steps:
            sample_path = results_dir / model_name / f"samples_steps_{step:03d}.npz"
            if not sample_path.is_file():
                raise FileNotFoundError(sample_path)
            sample_sets.append((step, _load_samples(sample_path, args.n_frames)))
        models.append({"name": model_name, "display_name": display_name, "sample_sets": sample_sets})

    reference_tica: np.ndarray | None = None
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None
    results: dict[str, Any] = {
        "data_path": str(args.data_path),
        "results_dir": str(results_dir),
        "sequence": sequence,
        "n_frames": int(reference_coords.shape[0]),
        "sample_steps": list(steps),
        "models": {},
    }

    for model in models:
        print(f"projecting {model['name']} to tICA")
        tica_eval = compute_tica_pmf_mjs(
            args.data_path,
            reference_coords,
            model["sample_sets"],
            sequence,
            n_bins=int(args.bins),
            tica_dim=2,
        )
        if reference_tica is None:
            reference_tica = np.asarray(tica_eval["reference_tica"], dtype=np.float64)
            limits = tica_eval["limits"]
            np.savez(out_dir / "reference_tica.npz", tica=reference_tica, limits=np.asarray(limits), data_path=str(args.data_path), sequence=sequence)
        model_tica_dir = out_dir / _safe_name(model["name"])
        model_tica_dir.mkdir(parents=True, exist_ok=True)
        model["tica_sets"] = []
        model["tica_metrics"] = {}
        model_results: dict[str, Any] = {"display_name": model["display_name"], "tica_paths": {}, "metrics": {}}
        for step, sample_tica in tica_eval["sample_sets"]:
            sample_tica = np.asarray(sample_tica, dtype=np.float64)
            tica_path = model_tica_dir / f"tica_steps_{step:03d}.npz"
            np.savez(tica_path, tica=sample_tica, sample_steps=int(step), limits=np.asarray(tica_eval["limits"]))
            w2 = _exact_tica_w2(reference_tica, sample_tica)
            metrics = {**tica_eval["metrics"][int(step)], "w2": w2, "n_reference_frames": int(reference_tica.shape[0]), "n_sample_frames": int(sample_tica.shape[0])}
            model["tica_sets"].append((int(step), sample_tica))
            model["tica_metrics"][int(step)] = metrics
            model_results["tica_paths"][str(int(step))] = str(tica_path)
            model_results["metrics"][str(int(step))] = metrics
            print(f"{model['name']} steps={step} tica_w2={w2:.6f} pmf={metrics['pmf']:.6f} mjs={metrics['mjs']:.6f} js={metrics['js']:.6f}")
        results["models"][model["name"]] = model_results

    if reference_tica is None or limits is None:
        raise RuntimeError("no tICA projections were computed")

    comparison_plots = plot_tica_comparison(
        reference_tica=reference_tica,
        models=models,
        limits=limits,
        out_path=out_dir / "tica_projection_comparison.png",
        bins=int(args.bins),
    )
    w2_plots = plot_tica_w2_by_steps(models=models, out_path=out_dir / "tica_w2_by_sample_steps.png")
    table_path = write_tica_w2_latex_table(models=models, out_path=out_dir / "tica_w2_table.tex")

    results["reference_tica_path"] = str(out_dir / "reference_tica.npz")
    results["limits"] = [[float(x) for x in pair] for pair in limits]
    results["comparison_plots"] = comparison_plots
    results["tica_w2_plots"] = w2_plots
    results["tica_w2_table"] = table_path

    results_path = out_dir / "tica_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed; disable --wandb or install wandb")
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config={"data_path": str(args.data_path), "sample_steps": list(steps), "n_frames": int(reference_coords.shape[0]), "bins": int(args.bins)})
        log_data = {}
        for model in models:
            for step, metrics in model["tica_metrics"].items():
                prefix = f"{model['name']}/tica_steps_{step}"
                log_data[f"{prefix}_w2"] = metrics["w2"]
                log_data[f"{prefix}_pmf"] = metrics["pmf"]
                log_data[f"{prefix}_mjs"] = metrics["mjs"]
                log_data[f"{prefix}_js"] = metrics["js"]
        wandb.log(log_data)
        artifact = wandb.Artifact(f"{_safe_name(args.wandb_name)}-tica-results", type="chignolin-tica-results")
        artifact.add_file(str(results_path))
        artifact.add_file(str(table_path))
        artifact.add_file(results["reference_tica_path"])
        for model_name, model_result in results["models"].items():
            for step, path in model_result["tica_paths"].items():
                artifact.add_file(path, name=f"{_safe_name(model_name)}/tica_steps_{int(step):03d}.npz")
        for plot_paths in (comparison_plots, w2_plots):
            for key in ("png", "pdf", "pgf"):
                if key in plot_paths:
                    artifact.add_file(plot_paths[key])
        wandb.log_artifact(artifact)
        run.finish()

    print(results_path)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sample-steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="Stochastic Flow Map")
    parser.add_argument("--wandb-name", default="chignolin-10k-tica-results")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
