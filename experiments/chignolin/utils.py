"""Shared utilities for Chignolin coordinate training experiments."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIE", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def workspace_root() -> Path:
    # Repo root: experiments/chignolin/utils.py -> parents[2]
    return Path(__file__).resolve().parents[2]


def resolve_out_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    if path.is_absolute():
        return path
    return workspace_root() / path


def run_output_dir(base_out_dir: str | Path, run_name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.=-]+", "-", run_name).strip("-._")
    if not safe_name:
        raise ValueError(f"invalid run name for output directory: {run_name!r}")
    return resolve_out_dir(base_out_dir) / safe_name


def ff14sb_xml_path() -> Path:
    return workspace_root() / "packages" / "thermax" / "assets" / "protein.ff14SB.xml"


def residue_atom_names(xml_path: Path) -> dict[str, list[str]]:
    root = ET.parse(xml_path).getroot()
    residues = root.find("Residues")
    if residues is None:
        raise ValueError(f"{xml_path} is missing Residues")
    return {
        residue.attrib["name"]: [atom.attrib["name"] for atom in residue.findall("Atom")]
        for residue in residues.findall("Residue")
    }


def residue_atom_elements(xml_path: Path) -> dict[str, list[str]]:
    root = ET.parse(xml_path).getroot()
    atom_types = root.find("AtomTypes")
    residues = root.find("Residues")
    if atom_types is None or residues is None:
        raise ValueError(f"{xml_path} is missing AtomTypes or Residues")
    type_to_element = {
        atom_type.attrib["name"]: atom_type.attrib["element"]
        for atom_type in atom_types.findall("Type")
    }
    return {
        residue.attrib["name"]: [type_to_element[atom.attrib["type"]] for atom in residue.findall("Atom")]
        for residue in residues.findall("Residue")
    }


def sequence_atom_elements(sequence: str) -> list[str]:
    templates = residue_atom_elements(ff14sb_xml_path())
    elements: list[str] = []
    for i, aa in enumerate(sequence):
        res = ONE_TO_THREE[aa]
        name = f"N{res}" if i == 0 else f"C{res}" if i == len(sequence) - 1 else res
        elements.extend(templates[name])
    return elements


def backbone_indices(sequence: str, n_atoms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    templates = residue_atom_names(ff14sb_xml_path())
    n_idx, ca_idx, c_idx, offset = [], [], [], 0
    for i, aa in enumerate(sequence):
        res = ONE_TO_THREE[aa]
        name = f"N{res}" if i == 0 else f"C{res}" if i == len(sequence) - 1 else res
        atoms = templates[name]
        n_idx.append(offset + atoms.index("N"))
        ca_idx.append(offset + atoms.index("CA"))
        c_idx.append(offset + atoms.index("C"))
        offset += len(atoms)
    if offset != n_atoms:
        raise ValueError(f"ff14SB atom count {offset} does not match positions atom count {n_atoms}")
    return np.asarray(n_idx), np.asarray(ca_idx), np.asarray(c_idx)


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    b0, b1, b2 = a - b, c - b, d - c
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def ramachandran(coords: np.ndarray, sequence: str) -> tuple[np.ndarray, np.ndarray]:
    features = ramachandran_features(coords, sequence)
    n_features = features.shape[1] // 2
    return features[:, :n_features].reshape(-1), features[:, n_features:].reshape(-1)


def ramachandran_features(coords: np.ndarray, sequence: str) -> np.ndarray:
    n_idx, ca_idx, c_idx = backbone_indices(sequence, coords.shape[1])
    phi = dihedral(coords[:, c_idx[:-1]], coords[:, n_idx[1:]], coords[:, ca_idx[1:]], coords[:, c_idx[1:]])
    psi = dihedral(coords[:, n_idx[:-1]], coords[:, ca_idx[:-1]], coords[:, c_idx[:-1]], coords[:, n_idx[1:]])
    return np.concatenate([phi, psi], axis=1)


def compute_torus_wasserstein(
    train_coords: np.ndarray,
    sample_sets: list[tuple[int, np.ndarray]],
    sequence: str,
) -> dict[str, Any]:
    """Compute W2 on the backbone phi/psi torus for each sample set."""
    from thermax_eval.metrics.torsion_w2 import torus_wasserstein

    reference_angles = ramachandran_features(train_coords, sequence)
    metrics: dict[int, dict[str, float | int]] = {}
    sample_angles = []
    for steps, coords in sample_sets:
        angles = ramachandran_features(coords, sequence)
        sample_angles.append((int(steps), angles))
        try:
            w2 = float(torus_wasserstein(angles, reference_angles))
        except Exception as exc:
            print(f"torus_wasserstein failed for steps={steps}: {exc}")
            w2 = float("nan")
        metrics[int(steps)] = {
            "w2": w2,
            "n_reference_frames": int(reference_angles.shape[0]),
            "n_sample_frames": int(angles.shape[0]),
            "n_torsion_features": int(reference_angles.shape[1]),
        }
    return {
        "reference_angles": reference_angles,
        "sample_sets": sample_angles,
        "metrics": metrics,
    }


def plot_torus_wasserstein(torus_eval: dict[str, Any], out_path: Path) -> None:
    """Plot torus Wasserstein distance versus sampler step count."""
    metrics = torus_eval["metrics"]
    steps = np.asarray(sorted(metrics), dtype=np.float64)
    values = np.asarray([metrics[int(step)]["w2"] for step in steps], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    finite = np.isfinite(values)
    if np.any(finite):
        ax.plot(steps[finite], values[finite], marker="o", linewidth=1.8)
    if np.any(~finite):
        ax.scatter(steps[~finite], np.zeros_like(steps[~finite]), marker="x", color="tab:red", label="failed")
        ax.legend(frameon=False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps)
    ax.set_xticklabels([str(int(step)) for step in steps], rotation=45, ha="right")
    ax.set_xlabel("sampling steps")
    ax.set_ylabel("torus W2 (rad)")
    ax.set_title("Backbone phi/psi torus Wasserstein")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def load_positions(path: Path, center_molecule: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        sequence = str(data["sequence"])
    if center_molecule:
        positions = positions - positions.mean(axis=1, keepdims=True)
    mean = positions.mean(axis=(0, 1), keepdims=True)
    std = positions.std(axis=(0, 1), keepdims=True).clip(1e-6)
    print(f"loaded {sequence}: positions={positions.shape}, center_molecule={center_molecule}, mean={mean.ravel()}, std={std.ravel()}")
    return (positions - mean) / std, mean.astype(np.float32), std.astype(np.float32), sequence


def vp_alpha_sigma(t: jnp.ndarray, beta_min: float, beta_max: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_alpha = -0.25 * (beta_max - beta_min) * t**2 - 0.5 * beta_min * t
    alpha = jnp.exp(log_alpha)
    sigma = jnp.sqrt(jnp.maximum(1.0 - alpha**2, 1e-5))
    return alpha, sigma


def eval_step_counts(max_steps: int) -> list[int]:
    steps = [max_steps]
    power = 1 << (max_steps.bit_length() - 1)
    if power == max_steps:
        power //= 2
    while power >= 1:
        steps.append(power)
        power //= 2
    return steps


def plot_ramachandran(
    train_coords: np.ndarray,
    sample_sets: list[tuple[int, np.ndarray]],
    sequence: str,
    out_path: Path,
    bins: int = 100,
) -> None:
    train_phi, train_psi = ramachandran(train_coords, sequence)
    panels = [(train_phi, train_psi, "data")]
    panels += [(*ramachandran(coords, sequence), f"{steps} steps") for steps, coords in sample_sets]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6), squeeze=False)
    for ax, (phi, psi, title) in zip(axes[0], panels, strict=True):
        ax.hist2d(phi, psi, bins=bins, range=[[-math.pi, math.pi], [-math.pi, math.pi]], norm=LogNorm(), rasterized=True)
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\varphi$")
        ax.set_ylabel(r"$\psi$")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _load_tica_reference(reference_npz_path: Path, sequence: str):
    from thermax_eval.many_peptides import load_reference_trajectory

    return load_reference_trajectory(reference_npz_path, sequence=sequence)


def _coords_to_zxyz(coords: np.ndarray, z: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (frames, atoms, 3); got {coords.shape}")
    if coords.shape[1] != z.shape[0]:
        raise ValueError(f"coords atom count {coords.shape[1]} does not match reference atom count {z.shape[0]}")
    samples = np.empty((coords.shape[0], coords.shape[1], 4), dtype=np.float64)
    samples[:, :, 0] = z[None, :]
    samples[:, :, 1:4] = coords
    return samples


def _project_tica_coords(coords: np.ndarray, reference: Any, tica_dim: int = 2) -> np.ndarray:
    from thermax_eval.targeting.featurization import compute_features

    dim = int(np.asarray(reference.tica.dim).item())
    tica_dim = min(int(tica_dim), dim)
    z = np.asarray(reference.samples[0, :, 0], dtype=np.float64)
    samples = _coords_to_zxyz(coords, z)
    features = compute_features(samples, reference.tica.featurization, smiles=reference.smiles)
    mean = np.asarray(reference.tica.mean, dtype=np.float64)
    projection = np.asarray(reference.tica.projection, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != mean.shape[0]:
        raise ValueError(f"feature shape {features.shape} does not match tICA mean shape {mean.shape}")
    if projection.ndim != 2 or projection.shape[0] != mean.shape[0] or tica_dim > projection.shape[1]:
        raise ValueError(f"projection shape {projection.shape} is incompatible with tICA dim {tica_dim}")
    return (features - mean[None, :]) @ projection[:, :tica_dim]


def _reference_tica_limits(
    reference_tica: np.ndarray,
    low_pct: float = 0.01,
    high_pct: float = 99.9,
    pad: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    reference_tica = np.asarray(reference_tica, dtype=np.float64)
    limits = []
    for axis in range(2):
        values = reference_tica[:, axis]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("reference tICA projection has no finite values")
        lo, hi = np.percentile(values, [low_pct, high_pct]).astype(np.float64)
        if lo == hi:
            delta = 1.0 if lo == 0.0 else abs(float(lo)) * 0.05
            lo -= delta
            hi += delta
        limits.append((float(lo - pad), float(hi + pad)))
    return limits[0], limits[1]


def _tica_histogram_prob(
    tica: np.ndarray,
    limits: tuple[tuple[float, float], tuple[float, float]],
    n_bins: int,
) -> np.ndarray:
    tica = np.asarray(tica, dtype=np.float64)
    finite = np.all(np.isfinite(tica[:, :2]), axis=1)
    hist, _, _ = np.histogram2d(
        tica[finite, 0],
        tica[finite, 1],
        bins=int(n_bins),
        range=[list(limits[0]), list(limits[1])],
    )
    total = float(hist.sum())
    if total <= 0.0:
        return np.zeros((int(n_bins), int(n_bins)), dtype=np.float64)
    return hist.astype(np.float64) / total


def _tica_grid_metrics(
    reference_tica: np.ndarray,
    sample_tica: np.ndarray,
    *,
    limits: tuple[tuple[float, float], tuple[float, float]],
    n_bins: int,
    baseline: float,
) -> dict[str, float]:
    ref = _tica_histogram_prob(reference_tica, limits, n_bins)
    sample = _tica_histogram_prob(sample_tica, limits, n_bins)
    if ref.sum() <= 0.0 or sample.sum() <= 0.0:
        return {"pmf": float("nan"), "mjs": float("nan"), "js": float("nan")}
    occupied = (ref > 0.0) | (sample > 0.0)
    p = np.where(ref[occupied] > 0.0, ref[occupied], baseline)
    q = np.where(sample[occupied] > 0.0, sample[occupied], baseline)
    weights = 0.5 * (p + q)
    weights /= weights.sum()

    pmf = float(np.sum(weights * np.square(-np.log(q) + np.log(p))))
    mixture = 0.5 * (p + q)
    mjs = float(np.sum(weights * 0.5 * ((p / mixture) * np.log(p / mixture) + (q / mixture) * np.log(q / mixture))))

    p_js = np.maximum(ref.ravel(), baseline)
    q_js = np.maximum(sample.ravel(), baseline)
    p_js /= p_js.sum()
    q_js /= q_js.sum()
    m_js = 0.5 * (p_js + q_js)
    js = float(0.5 * np.sum(p_js * np.log(p_js / m_js)) + 0.5 * np.sum(q_js * np.log(q_js / m_js)))
    return {"pmf": pmf, "mjs": mjs, "js": js}


def compute_tica_pmf_mjs(
    reference_npz_path: Path,
    train_coords: np.ndarray,
    sample_sets: list[tuple[int, np.ndarray]],
    sequence: str,
    *,
    n_bins: int = 100,
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    baseline: float = 1e-6,
    tica_dim: int = 2,
) -> dict[str, Any]:
    """Project Chignolin samples to reference tICA space and compute 2D grid metrics.

    The reference NPZ is expected to contain many-peptides ``tica_mean``,
    ``tica_projection``, and ``tica_dim`` arrays. Missing ``tica_featurization``
    follows thermax-eval's default of ``cns_dihedrals``.
    """
    reference = _load_tica_reference(Path(reference_npz_path), sequence)
    reference_tica = _project_tica_coords(train_coords, reference, tica_dim=tica_dim)
    if reference_tica.shape[1] < 2:
        raise ValueError(f"need at least two tICA dimensions for 2D metrics; got {reference_tica.shape[1]}")
    projected_sample_sets = []
    for steps, coords in sample_sets:
        sample_tica = _project_tica_coords(coords, reference, tica_dim=tica_dim)
        projected_sample_sets.append((int(steps), sample_tica))

    metric_limits = _reference_tica_limits(reference_tica) if limits is None else limits

    metrics: dict[int, dict[str, float]] = {}
    for steps, sample_tica in projected_sample_sets:
        metrics[int(steps)] = _tica_grid_metrics(
            reference_tica,
            sample_tica,
            limits=metric_limits,
            n_bins=int(n_bins),
            baseline=float(baseline),
        )
    return {
        "reference_tica": reference_tica,
        "sample_sets": projected_sample_sets,
        "metrics": metrics,
        "limits": metric_limits,
        "n_bins": int(n_bins),
        "baseline": float(baseline),
    }


def plot_tica_projections(
    tica_eval: dict[str, Any],
    out_path: Path,
    *,
    bins: int | None = None,
) -> None:
    """Plot reference and k-step sample densities in the computed tICA plane."""
    reference_tica = np.asarray(tica_eval["reference_tica"], dtype=np.float64)
    sample_sets = tica_eval["sample_sets"]
    limits = tica_eval["limits"]
    bins = int(tica_eval.get("n_bins", 100) if bins is None else bins)
    panels = [(reference_tica, "data")]
    panels += [(np.asarray(tica, dtype=np.float64), f"{steps} steps") for steps, tica in sample_sets]

    positive_hist_values = []
    for tica, _ in panels:
        finite = np.all(np.isfinite(tica[:, :2]), axis=1)
        hist, _, _ = np.histogram2d(
            tica[finite, 0],
            tica[finite, 1],
            bins=bins,
            range=[list(limits[0]), list(limits[1])],
            density=True,
        )
        positive_hist_values.append(hist[hist > 0.0])
    positive = np.concatenate([x for x in positive_hist_values if x.size > 0]) if any(x.size > 0 for x in positive_hist_values) else np.asarray([1.0])
    norm = LogNorm(vmin=max(float(positive.min()), 1e-2), vmax=max(float(positive.max()), 1e-1))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.6), squeeze=False)
    for ax, (tica, title) in zip(axes[0], panels, strict=True):
        finite = np.all(np.isfinite(tica[:, :2]), axis=1)
        ax.hist2d(
            tica[finite, 0],
            tica[finite, 1],
            bins=bins,
            range=[list(limits[0]), list(limits[1])],
            norm=norm,
            rasterized=True,
        )
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        ax.set_aspect("auto")
        ax.set_xlabel("TIC 0")
        ax.set_ylabel("TIC 1")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
