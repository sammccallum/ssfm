"""Export random XYZ conformers from saved Chignolin sample NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import sequence_atom_elements


DEFAULT_RESULTS_DIR = Path("experiments/chignolin/runs/results")
DEFAULT_SEQUENCE = "GYDPETGTWG"
DEFAULT_MODELS = ("xpred-diffusion", "strong-sfm-eta0.75")
DEFAULT_STEPS = (100, 64, 32, 16, 8, 4, 2, 1)


def write_xyz(path: Path, coords: np.ndarray, elements: list[str], comment: str, scale: float) -> str:
    coords = np.asarray(coords, dtype=np.float64) * float(scale)
    if coords.shape != (len(elements), 3):
        raise ValueError(f"coords shape {coords.shape} does not match {len(elements)} elements")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(elements)), comment]
    lines.extend(
        f"{element:2s} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}"
        for element, xyz in zip(elements, coords, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def export_xyz_samples(
    *,
    results_dir: Path,
    sequence: str,
    models: tuple[str, ...],
    steps: tuple[int, ...],
    n_per_step: int,
    seed: int,
    scale: float,
) -> dict[str, dict[str, list[str]]]:
    elements = sequence_atom_elements(sequence)
    rng = np.random.default_rng(seed)
    out_dir = results_dir / "xyz_samples"
    exported: dict[str, dict[str, list[str]]] = {}

    for model in models:
        exported[model] = {}
        for step in steps:
            npz_path = results_dir / model / f"samples_steps_{step:03d}.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(npz_path)
            with np.load(npz_path, allow_pickle=True) as data:
                positions = np.asarray(data["positions"], dtype=np.float64)
            frame_indices = rng.choice(positions.shape[0], size=n_per_step, replace=positions.shape[0] < n_per_step)
            paths = []
            for export_idx, frame_idx in enumerate(frame_indices, start=1):
                xyz_path = out_dir / model / f"steps_{step:03d}" / f"sample_{export_idx:02d}.xyz"
                paths.append(
                    write_xyz(
                        xyz_path,
                        positions[int(frame_idx)],
                        elements,
                        f"{model}; steps={step}; frame={int(frame_idx)}; units=angstrom; scale={scale:g}; source={npz_path}",
                        scale,
                    )
                )
            exported[model][str(step)] = paths

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(exported, indent=2), encoding="utf-8")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--n-per-step", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--scale", type=float, default=10.0, help="Coordinate scale before writing XYZ; default converts nm to Angstrom.")
    args = parser.parse_args()

    exported = export_xyz_samples(
        results_dir=args.results_dir,
        sequence=args.sequence,
        models=tuple(args.models),
        steps=tuple(args.steps),
        n_per_step=int(args.n_per_step),
        seed=int(args.seed),
        scale=float(args.scale),
    )
    print(json.dumps(exported, indent=2))


if __name__ == "__main__":
    main()
