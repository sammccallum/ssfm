"""Shared Chignolin checkpoint sampling and figure I/O helpers.

Both ``torus_results.py`` (torus-W2 + Ramachandran) and ``tica_results.py`` (tICA
projection + W2/PMF/JS) build on this sampling stage so the 10k conformer NPZs
are produced and named identically.
"""

from __future__ import annotations

import importlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import jax
import numpy as np

MODEL_MODULES = {
    "xpred": "train_xpred_diffusion",
    "strong": "train_strong_sfm",
}


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "-", name).strip("-._")
    if not safe:
        raise ValueError(f"invalid output name {name!r}")
    return safe


def _step_from_checkpoint_path(path: Path) -> int | None:
    if path.stem.startswith("step_"):
        return int(path.stem.removeprefix("step_"))
    return None


def _resolve_checkpoint(path: Path) -> tuple[Path, dict[str, Any]]:
    """Return an existing checkpoint, falling back to nearest step in the same dir."""
    path = path.expanduser()
    if path.exists():
        return path, {"requested": str(path), "resolved": str(path), "resolved_reason": "exact"}

    requested_step = _step_from_checkpoint_path(path)
    candidates = sorted(path.parent.glob("step_*.pkl"))
    if requested_step is None or not candidates:
        raise FileNotFoundError(f"checkpoint not found: {path}")

    def distance(candidate: Path) -> tuple[int, int]:
        step = _step_from_checkpoint_path(candidate)
        if step is None:
            return (10**18, 10**18)
        return (abs(step - requested_step), step)

    resolved = min(candidates, key=distance)
    return resolved, {
        "requested": str(path),
        "resolved": str(resolved),
        "resolved_reason": f"nearest_available_to_{requested_step}",
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    config_path = path.parents[1] / "config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            ckpt["metadata"] = json.load(f)
    return ckpt


def _metadata_value(ckpt: dict[str, Any], name: str, default: Any) -> Any:
    if name in ckpt:
        return ckpt[name]
    metadata = ckpt.get("metadata")
    if isinstance(metadata, dict):
        training = metadata.get("training")
        if isinstance(training, dict) and name in training:
            return training[name]
    return default


def _sample_checkpoint(
    *,
    model_kind: str,
    ckpt: dict[str, Any],
    n_samples: int,
    n_sample_steps: int,
    key: jax.Array,
) -> np.ndarray:
    module = importlib.import_module(MODEL_MODULES[model_kind])
    params = ckpt.get("ema_params", ckpt["params"])
    config = ckpt["config"]

    if model_kind == "xpred":
        samples = module.sample(
            params,
            config,
            n_samples,
            n_sample_steps,
            key,
            float(_metadata_value(ckpt, "beta_min", 0.1)),
            float(_metadata_value(ckpt, "beta_max", 20.0)),
        )
    elif model_kind == "strong":
        samples = module.sample(
            params,
            config,
            n_samples,
            n_sample_steps,
            key,
            float(_metadata_value(ckpt, "t_eps", 1e-4)),
            bool(_metadata_value(ckpt, "pass_brownian_coefficients", True)),
            bool(_metadata_value(ckpt, "pass_target_time", True)),
            bool(_metadata_value(ckpt, "pass_any_time", True)),
        )
    else:
        raise ValueError(f"unknown model kind {model_kind!r}")

    return np.asarray(samples) * np.asarray(ckpt["std"]) + np.asarray(ckpt["mean"])


def _sample_checkpoint_batched(
    *,
    model_kind: str,
    ckpt: dict[str, Any],
    n_samples: int,
    n_sample_steps: int,
    key: jax.Array,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0 or batch_size >= n_samples:
        return _sample_checkpoint(
            model_kind=model_kind,
            ckpt=ckpt,
            n_samples=n_samples,
            n_sample_steps=n_sample_steps,
            key=key,
        )

    chunks = []
    remaining = int(n_samples)
    while remaining > 0:
        key, key_chunk = jax.random.split(key)
        chunk_size = min(int(batch_size), remaining)
        chunks.append(
            _sample_checkpoint(
                model_kind=model_kind,
                ckpt=ckpt,
                n_samples=chunk_size,
                n_sample_steps=n_sample_steps,
                key=key_chunk,
            )
        )
        remaining -= chunk_size
    return np.concatenate(chunks, axis=0)


def find_existing_samples(run_dir: Path, sample_steps: tuple[int, ...]) -> dict[int, Path]:
    found = {}
    for n_steps in sample_steps:
        path = run_dir / f"latest_samples_steps_{n_steps:03d}.npz"
        if path.is_file():
            found[n_steps] = path
    return found


def _load_existing_samples(path: Path, n_samples: int) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        positions = np.asarray(data["positions"], dtype=np.float64)
    if positions.shape[0] < n_samples:
        raise ValueError(f"{path} has only {positions.shape[0]} samples, need {n_samples}")
    return positions[:n_samples]


def sample_or_reuse_model(
    *,
    name: str,
    model_kind: str,
    checkpoint_path: Path,
    out_dir: Path,
    n_samples: int,
    sample_steps: tuple[int, ...],
    seed: int,
    reuse_existing: bool,
    sample_batch_size: int,
) -> dict[str, Any]:
    checkpoint_path, checkpoint_resolution = _resolve_checkpoint(checkpoint_path)
    ckpt = _load_checkpoint(checkpoint_path)
    model_dir = out_dir / _safe_name(name)
    model_dir.mkdir(parents=True, exist_ok=True)
    existing = find_existing_samples(checkpoint_path.parents[1], sample_steps)

    key = jax.random.PRNGKey(seed)
    sample_sets = []
    sample_paths = {}
    reused_paths = {}
    for n_steps in sample_steps:
        out_path = model_dir / f"samples_steps_{n_steps:03d}.npz"
        if reuse_existing and out_path.is_file():
            samples_np = _load_existing_samples(out_path, n_samples)
            reused_paths[str(n_steps)] = str(out_path)
        elif reuse_existing and n_steps in existing:
            samples_np = _load_existing_samples(existing[n_steps], n_samples)
            reused_paths[str(n_steps)] = str(existing[n_steps])
        else:
            key, key_sample = jax.random.split(key)
            samples_np = _sample_checkpoint_batched(
                model_kind=model_kind,
                ckpt=ckpt,
                n_samples=n_samples,
                n_sample_steps=n_steps,
                key=key_sample,
                batch_size=sample_batch_size,
            )
        np.savez(
            out_path,
            positions=samples_np.astype(np.float32),
            sample_steps=n_steps,
            checkpoint=str(checkpoint_path),
            model_kind=model_kind,
            reused_from=reused_paths.get(str(n_steps), ""),
        )
        sample_sets.append((n_steps, samples_np))
        sample_paths[str(n_steps)] = str(out_path)

    return {
        "name": name,
        "model_kind": model_kind,
        "checkpoint": checkpoint_resolution,
        "checkpoint_step": int(ckpt.get("step", _step_from_checkpoint_path(checkpoint_path) or -1)),
        "center_molecule": bool(ckpt.get("center_molecule", False)),
        "sample_sets": sample_sets,
        "sample_paths": sample_paths,
        "reused_sample_paths": reused_paths,
        "available_existing_sample_paths": {str(k): str(v) for k, v in existing.items()},
    }


def save_figure(fig, out_path: Path, *, dpi: int = 180) -> dict[str, Any]:
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


# Backwards-compatible alias for the private name used internally before the split.
_save_figure = save_figure
