"""Aggregate Chignolin checkpoint samples into torus-W2 metrics and figures."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from omegaconf import DictConfig, OmegaConf

from sampling import (
    _resolve_checkpoint,
    _safe_name,
    _save_figure,
    sample_or_reuse_model,
)
from utils import compute_torus_wasserstein, load_positions, ramachandran, ramachandran_features, sequence_atom_elements

try:
    import wandb
except ImportError:  # pragma: no cover - optional logging dependency.
    wandb = None


DEFAULT_DATA_PATH = Path("/mnt/labs/data/tong/many-peptides-md/trajectories_subsampled/test/10AA/GYDPETGTWG_subsampled.npz")
DEFAULT_OUT_DIR = Path("experiments/chignolin/runs/results")
DEFAULT_XPRED_CHECKPOINT = Path(
    "experiments/chignolin/runs/xpred_diffusion/checkpoints/latest.pkl"
)
DEFAULT_SSF_CHECKPOINT = Path(
    "experiments/chignolin/runs/strong_sfm/chignolin-strong-sfm-eta0.75/checkpoints/latest.pkl"
)
DEFAULT_SAMPLE_STEPS = (100, 32, 8, 2)


def _panel_hist2d(ax, phi: np.ndarray, psi: np.ndarray, title: str, bins: int) -> None:
    ax.hist2d(
        phi,
        psi,
        bins=bins,
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
        norm=LogNorm(),
        rasterized=True,
    )
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([-np.pi, 0.0, np.pi])
    ax.set_yticks([-np.pi, 0.0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=7)
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=7)


def _save_figure(fig, out_path: Path, *, dpi: int = 180) -> dict[str, Any]:
    """Save a figure in raster and TeX-friendly vector formats."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = out_path.name[: -len(out_path.suffix)] if out_path.suffix else out_path.name
    paths: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for fmt in ("png", "pdf"):
        path = out_path.parent / f"{base_name}.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths[fmt] = str(path)

    pgf_path = out_path.parent / f"{base_name}.pgf"
    try:
        fig.savefig(pgf_path)
        paths["pgf"] = str(pgf_path)
    except Exception as exc:  # pragma: no cover - depends on local TeX install.
        errors["pgf"] = f"{type(exc).__name__}: {exc}"
        print(f"failed to save PGF figure {pgf_path}: {errors['pgf']}")

    if errors:
        paths["errors"] = errors
    return paths


def write_xyz(path: Path, coords: np.ndarray, elements: list[str], comment: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords, dtype=np.float64) * 10.0
    if coords.shape != (len(elements), 3):
        raise ValueError(f"coords shape {coords.shape} does not match {len(elements)} elements")
    lines = [str(len(elements)), comment]
    lines.extend(
        f"{element:2s} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}"
        for element, xyz in zip(elements, coords, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def export_random_xyzs(
    *,
    models: list[dict[str, Any]],
    sequence: str,
    out_dir: Path,
    n_per_step: int,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    elements = sequence_atom_elements(sequence)
    rng = np.random.default_rng(seed)
    exported: dict[str, dict[str, list[str]]] = {}
    for model in models:
        model_name = _safe_name(model["name"])
        exported[model["name"]] = {}
        for steps, coords in model["sample_sets"]:
            coords = np.asarray(coords, dtype=np.float64)
            n_frames = int(coords.shape[0])
            replace = n_frames < n_per_step
            frame_indices = rng.choice(n_frames, size=n_per_step, replace=replace)
            step_paths = []
            for export_idx, frame_idx in enumerate(frame_indices, start=1):
                path = out_dir / model_name / f"steps_{int(steps):03d}" / f"sample_{export_idx:02d}.xyz"
                step_paths.append(
                    write_xyz(
                        path,
                        coords[int(frame_idx)],
                        elements,
                        f"{model.get('display_name', model['name'])}; steps={int(steps)}; frame={int(frame_idx)}; units=angstrom; scale=10",
                    )
                )
            exported[model["name"]][str(int(steps))] = step_paths
    return exported


def plot_per_torsion_grid(
    *,
    reference_coords: np.ndarray,
    sample_sets: list[tuple[int, np.ndarray]],
    sequence: str,
    out_path: Path,
    bins: int,
) -> dict[str, Any]:
    rows = [("data", reference_coords)] + [(f"{steps} steps", coords) for steps, coords in sample_sets]
    reference_angles = ramachandran_features(reference_coords, sequence)
    n_pairs = reference_angles.shape[1] // 2

    fig, axes = plt.subplots(len(rows), n_pairs, figsize=(2.1 * n_pairs, 2.0 * len(rows)), squeeze=False)
    for row_idx, (row_label, coords) in enumerate(rows):
        angles = ramachandran_features(coords, sequence)
        phi = angles[:, :n_pairs]
        psi = angles[:, n_pairs:]
        for torsion_idx in range(n_pairs):
            title = f"res {torsion_idx + 2}" if row_idx == 0 else ""
            _panel_hist2d(axes[row_idx, torsion_idx], phi[:, torsion_idx], psi[:, torsion_idx], title, bins)
            if torsion_idx == 0:
                axes[row_idx, torsion_idx].set_ylabel(row_label, fontsize=9)
    fig.tight_layout()
    paths = _save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return paths


def plot_pooled_comparison(
    *,
    reference_coords: np.ndarray,
    models: list[dict[str, Any]],
    sequence: str,
    out_path: Path,
    bins: int,
) -> dict[str, Any]:
    column_steps = [steps for steps, _ in models[0]["sample_sets"]]
    columns = ["data"] + [f"{steps} steps" for steps in column_steps]
    fig, axes = plt.subplots(len(models), len(columns), figsize=(3.0 * len(columns), 2.8 * len(models)), squeeze=False)

    data_phi, data_psi = ramachandran(reference_coords, sequence)
    for row_idx, model in enumerate(models):
        sample_by_step = {steps: coords for steps, coords in model["sample_sets"]}
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                phi, psi = data_phi, data_psi
            else:
                step = column_steps[col_idx - 1]
                phi, psi = ramachandran(sample_by_step[step], sequence)
            _panel_hist2d(ax, phi, psi, column if row_idx == 0 else "", bins)
            if col_idx == 0:
                ax.set_ylabel(model.get("display_name", model["name"]), fontsize=9)
    fig.tight_layout()
    paths = _save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return paths


def plot_torus_w2_by_steps(
    *,
    models: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for model in models:
        metrics = model["torus_w2"]
        steps = sorted(int(step) for step in metrics)
        values = np.asarray([float(metrics[step]["w2"]) for step in steps], dtype=np.float64)
        finite = np.isfinite(values)
        label = model.get("display_name", model["name"])
        if finite.any():
            ax.plot(np.asarray(steps)[finite], values[finite], marker="o", linewidth=1.8, label=label)
        if (~finite).any():
            ax.scatter(np.asarray(steps)[~finite], np.zeros((~finite).sum()), marker="x", color="tab:red")

    ax.set_xscale("log", base=2)
    all_steps = sorted({int(step) for model in models for step in model["torus_w2"]})
    ax.set_xticks(all_steps)
    ax.set_xticklabels([str(step) for step in all_steps])
    ax.set_xlabel("sample steps")
    ax.set_ylabel("torus W2 (rad)")
    ax.set_title("Backbone phi/psi torus Wasserstein")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = _save_figure(fig, out_path, dpi=180)
    plt.close(fig)
    return paths


def write_torus_w2_latex_table(
    *,
    models: list[dict[str, Any]],
    out_path: Path,
    precision: int = 3,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_steps = sorted({int(step) for model in models for step in model["torus_w2"]})
    columns = "ll" + "r" * len(sample_steps)
    nfe_start = 3
    nfe_end = nfe_start + len(sample_steps) - 1
    best_by_step = {
        step: min(float(model["torus_w2"][step]["w2"]) for model in models if np.isfinite(float(model["torus_w2"][step]["w2"])))
        for step in sample_steps
    }
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Backbone torsion $\mathbb{T}$-W$_2$ ($\downarrow$) on Chignolin across step counts.}",
        r"  \label{tab:chignolin_torus_w2}",
        r"  \begin{tabular}{" + columns + "}",
        r"  \toprule",
        rf"   & & \multicolumn{{{len(sample_steps)}}}{{c}}{{NFE}} \\",
        rf"  \cmidrule(lr){{{nfe_start}-{nfe_end}}}",
        "   Metric & Method & " + " & ".join(str(step) for step in sample_steps) + r" \\",
        r"  \midrule",
    ]
    for idx, model in enumerate(models):
        metrics = model["torus_w2"]
        values = []
        for step in sample_steps:
            value = float(metrics[step]["w2"])
            if not np.isfinite(value):
                values.append(r"\mathrm{nan}")
            elif np.isclose(value, best_by_step[step]):
                values.append(rf"\textbf{{{value:.{precision}f}}}")
            else:
                values.append(f"{value:.{precision}f}")
        metric = r"\multirow{2}{*}{$\mathbb{T}$-W$_2$}" if idx == 0 else ""
        method = model.get("display_name", model["name"])
        if method == "SSFM":
            method = r"\textbf{SSFM}"
        lines.append("   " + metric + " & " + method + " & " + " & ".join(values) + r" \\")
    lines.extend([r"  \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    table = "\n".join(lines)
    out_path.write_text(table, encoding="utf-8")
    return str(out_path)


def run(args: argparse.Namespace | SimpleNamespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_steps = tuple(int(step) for step in args.sample_steps)

    xpred_checkpoint, _ = _resolve_checkpoint(args.xpred_checkpoint)
    ssfm_checkpoint, _ = _resolve_checkpoint(args.ssfm_checkpoint)
    xpred_ckpt = _load_checkpoint(xpred_checkpoint)
    ssfm_ckpt = _load_checkpoint(ssfm_checkpoint)
    if bool(xpred_ckpt.get("center_molecule", False)) != bool(ssfm_ckpt.get("center_molecule", False)):
        raise ValueError("xpred and SSFM checkpoints disagree on center_molecule")

    train_np, mean, std, sequence = load_positions(args.data_path, bool(xpred_ckpt.get("center_molecule", False)))
    reference_coords = train_np[: args.n_samples] * std + mean

    wandb_run = None
    if bool(getattr(args, "wandb", False)):
        if wandb is None:
            raise ImportError("wandb is not installed; disable wandb logging or install wandb")
        wandb_run = wandb.init(
            project=str(args.wandb_project),
            name=str(args.wandb_name),
            config={
                "data_path": str(args.data_path),
                "out_dir": str(args.out_dir),
                "xpred_checkpoint": str(args.xpred_checkpoint),
                "ssfm_checkpoint": str(args.ssfm_checkpoint),
                "n_samples": int(args.n_samples),
                "sample_steps": list(sample_steps),
                "seed": int(args.seed),
                "bins": int(args.bins),
                "reuse_existing_samples": bool(args.reuse_existing_samples),
                "sample_batch_size": int(args.sample_batch_size),
            },
        )

    models = [
        sample_or_reuse_model(
            name="xpred-diffusion",
            model_kind="xpred",
            checkpoint_path=args.xpred_checkpoint,
            out_dir=args.out_dir,
            n_samples=args.n_samples,
            sample_steps=sample_steps,
            seed=args.seed,
            reuse_existing=args.reuse_existing_samples,
            sample_batch_size=args.sample_batch_size,
        ),
        sample_or_reuse_model(
            name="strong-sfm-eta0.75",
            model_kind="strong",
            checkpoint_path=args.ssfm_checkpoint,
            out_dir=args.out_dir,
            n_samples=args.n_samples,
            sample_steps=sample_steps,
            seed=args.seed + 1,
            reuse_existing=args.reuse_existing_samples,
            sample_batch_size=args.sample_batch_size,
        ),
    ]
    models[0]["display_name"] = "Diffusion"
    models[1]["display_name"] = "SSFM"

    reference_path = args.out_dir / "reference_samples.npz"
    np.savez(reference_path, positions=reference_coords.astype(np.float32), data_path=str(args.data_path), sequence=sequence)

    results: dict[str, Any] = {
        "data_path": str(args.data_path),
        "reference_path": str(reference_path),
        "sequence": sequence,
        "n_samples": args.n_samples,
        "sample_steps": list(sample_steps),
        "reuse_existing_samples": bool(args.reuse_existing_samples),
        "models": {},
    }

    for model in models:
        torus_eval = compute_torus_wasserstein(reference_coords, model["sample_sets"], sequence)
        model["torus_w2"] = torus_eval["metrics"]
        per_torsion_path = args.out_dir / f"{_safe_name(model['name'])}_per_torsion_ramachandran.png"
        per_torsion_plots = plot_per_torsion_grid(
            reference_coords=reference_coords,
            sample_sets=model["sample_sets"],
            sequence=sequence,
            out_path=per_torsion_path,
            bins=args.bins,
        )
        results["models"][model["name"]] = {
            "display_name": model["display_name"],
            "model_kind": model["model_kind"],
            "checkpoint": model["checkpoint"],
            "checkpoint_step": model["checkpoint_step"],
            "sample_paths": model["sample_paths"],
            "reused_sample_paths": model["reused_sample_paths"],
            "available_existing_sample_paths": model["available_existing_sample_paths"],
            "torus_w2": {str(k): v for k, v in torus_eval["metrics"].items()},
            "per_torsion_plot": str(per_torsion_path),
            "per_torsion_plots": per_torsion_plots,
        }
        if wandb_run is not None:
            wandb.log(
                {
                    f"{model['name']}/torus_w2_steps_{steps}": metrics["w2"]
                    for steps, metrics in torus_eval["metrics"].items()
                }
            )

    pooled_path = args.out_dir / "pooled_ramachandran_comparison.png"
    pooled_plots = plot_pooled_comparison(
        reference_coords=reference_coords,
        models=models,
        sequence=sequence,
        out_path=pooled_path,
        bins=args.bins,
    )
    results["pooled_plot"] = str(pooled_path)
    results["pooled_plots"] = pooled_plots

    torus_w2_path = args.out_dir / "torus_w2_by_sample_steps.png"
    torus_w2_plots = plot_torus_w2_by_steps(models=models, out_path=torus_w2_path)
    results["torus_w2_plot"] = str(torus_w2_path)
    results["torus_w2_plots"] = torus_w2_plots

    torus_w2_table_path = args.out_dir / "torus_w2_table.tex"
    results["torus_w2_table"] = write_torus_w2_latex_table(models=models, out_path=torus_w2_table_path)

    xyz_dir = args.out_dir / "xyz_samples"
    results["xyz_samples"] = export_random_xyzs(
        models=models,
        sequence=sequence,
        out_dir=xyz_dir,
        n_per_step=int(args.xyz_samples_per_step),
        seed=int(args.seed) + 20260530,
    )

    results_path = args.out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if wandb_run is not None:
        artifact = wandb.Artifact(f"{_safe_name(args.wandb_name)}-results", type="chignolin-results")
        artifact.add_file(str(results_path))
        for model in results["models"].values():
            for path in model["sample_paths"].values():
                artifact.add_file(path)
            plot_paths = model.get("per_torsion_plots", {})
            for key in ("png", "pdf", "pgf"):
                if key in plot_paths:
                    artifact.add_file(plot_paths[key])
        for plot_paths in (pooled_plots, torus_w2_plots):
            for key in ("png", "pdf", "pgf"):
                if key in plot_paths:
                    artifact.add_file(plot_paths[key])
        artifact.add_file(results["torus_w2_table"])
        for model_paths in results["xyz_samples"].values():
            for step_paths in model_paths.values():
                for path in step_paths:
                    artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb_run.finish()
    print(results_path)
    print(json.dumps(results, indent=2))


def cli_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--xpred-checkpoint", type=Path, default=DEFAULT_XPRED_CHECKPOINT)
    parser.add_argument("--ssfm-checkpoint", type=Path, default=DEFAULT_SSF_CHECKPOINT)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--sample-steps", type=int, nargs="+", default=list(DEFAULT_SAMPLE_STEPS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--reuse-existing-samples", action="store_true")
    parser.add_argument("--sample-batch-size", type=int, default=512)
    parser.add_argument("--xyz-samples-per-step", type=int, default=3)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="Stochastic Flow Map")
    parser.add_argument("--wandb-name", default="chignolin-10k-results")
    run(parser.parse_args())


@hydra.main(version_base="1.3", config_path="../../configs", config_name="chignolin_results")
def main(cfg: DictConfig) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("resolved Hydra config must be a dict")
    run(
        SimpleNamespace(
            data_path=Path(str(resolved["data_path"])),
            out_dir=Path(str(resolved["out_dir"])),
            xpred_checkpoint=Path(str(resolved["xpred_checkpoint"])),
            ssfm_checkpoint=Path(str(resolved["ssfm_checkpoint"])),
            n_samples=int(resolved["n_samples"]),
            sample_steps=[int(step) for step in resolved["sample_steps"]],
            seed=int(resolved["seed"]),
            bins=int(resolved["bins"]),
            reuse_existing_samples=bool(resolved["reuse_existing_samples"]),
            sample_batch_size=int(resolved["sample_batch_size"]),
            xyz_samples_per_step=int(resolved["xyz_samples_per_step"]),
            wandb=bool(resolved["wandb"]["enabled"]),
            wandb_project=str(resolved["wandb"]["project"]),
            wandb_name=str(resolved["wandb"]["name"]),
        )
    )


if __name__ == "__main__":
    main()
