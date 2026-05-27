"""Train an x-prediction VP diffusion baseline on Chignolin coordinates.

Defaults to:
DATA_BASE=/mnt/labs/data/tong
DATA_REL=many-peptides-md/trajectories_subsampled/test/10AA/GYDPETGTWG_subsampled.npz
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ssfm.dit import apply_dit, init_dit

from utils import (
    compute_tica_pmf_mjs,
    compute_torus_wasserstein,
    eval_step_counts,
    load_positions,
    plot_ramachandran,
    plot_tica_projections,
    plot_torus_wasserstein,
    resolve_out_dir,
    run_output_dir,
    vp_alpha_sigma,
)


def make_train_step(model_config: dict[str, Any], optimizer: optax.GradientTransformation, beta_min: float, beta_max: float):
    @jax.jit
    def train_step(params: dict[str, Any], opt_state: optax.OptState, batch: jnp.ndarray, key: jnp.ndarray):
        key_s, key_noise, key_model = jax.random.split(key, 3)
        s = jax.random.uniform(key_s, (batch.shape[0],), minval=1e-4, maxval=1.0)
        noise = jax.random.normal(key_noise, batch.shape)
        alpha_s, sigma_s = vp_alpha_sigma(s[:, None, None], beta_min, beta_max)
        x_s = alpha_s * batch + sigma_s * noise

        def loss_fn(p):
            keys = jax.random.split(key_model, batch.shape[0])
            pred = jax.vmap(lambda x, si, k: apply_dit(p, model_config, x, si, key=k))(x_s, s, keys)
            return jnp.mean(jnp.square(pred - batch))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return train_step


def sample(params: dict[str, Any], config: dict[str, Any], n_samples: int, n_steps: int, key: jnp.ndarray, beta_min: float, beta_max: float) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n_samples, config["n_atoms"], config["coord_dim"]))
    times = jnp.linspace(1.0, 1e-3, n_steps)
    for t in times:
        key, key_model, key_noise = jax.random.split(key, 3)
        pred_x0 = jax.vmap(lambda y, k: apply_dit(params, config, y, t, key=k))(x, jax.random.split(key_model, n_samples))
        s = jnp.maximum(t - 1.0 / n_steps, 1e-3)
        alpha_t, sigma_t = vp_alpha_sigma(t, beta_min, beta_max)
        alpha_s, sigma_s = vp_alpha_sigma(s, beta_min, beta_max)
        alpha_t_given_s = alpha_t / alpha_s
        sigma_t_given_s2 = jnp.maximum(sigma_t**2 - alpha_t_given_s**2 * sigma_s**2, 1e-12)
        mean = (alpha_t_given_s * sigma_s**2 / sigma_t**2) * x
        mean = mean + (alpha_s * sigma_t_given_s2 / sigma_t**2) * pred_x0
        var = jnp.maximum(sigma_s**2 * sigma_t_given_s2 / sigma_t**2, 0.0)
        noise = jax.random.normal(key_noise, x.shape)
        x = mean + jnp.sqrt(var) * noise * (s > 1e-3)
    return x


def save_checkpoint(path: Path, params: dict[str, Any], config: dict[str, Any], opt_state: optax.OptState, mean: np.ndarray, std: np.ndarray, sequence: str, center_molecule: bool, step: int, key: jnp.ndarray, beta_min: float, beta_max: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"params": params, "config": config, "opt_state": opt_state, "mean": mean, "std": std, "sequence": sequence, "center_molecule": center_molecule, "step": step, "key": key, "beta_min": beta_min, "beta_max": beta_max}, f)


def train(cfg: DictConfig) -> None:
    out_dir = run_output_dir(cfg.out_dir, str(cfg.wandb.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(cfg.data_base) / cfg.data_rel
    train_np, mean, std, sequence = load_positions(data_path, bool(cfg.center_molecule))
    n_atoms = train_np.shape[1]

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    metadata = cfg_dict | {
        "base_out_dir": str(resolve_out_dir(cfg.out_dir)),
        "resolved_out_dir": str(out_dir),
        "data_path": str(data_path),
        "sequence": sequence,
        "n_atoms": int(n_atoms),
        "n_frames": int(train_np.shape[0]),
        "hydra_output_dir": HydraConfig.get().runtime.output_dir if HydraConfig.initialized() else str(Path.cwd()),
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    wandb_run = None
    if bool(cfg.wandb.enabled):
        wandb_run = wandb.init(
            project=str(cfg.wandb.project),
            name=str(cfg.wandb.name),
            config=metadata,
        )

    key = jax.random.PRNGKey(int(cfg.seed))
    key, key_model = jax.random.split(key)
    params, model_config = init_dit(
        key_model,
        n_atoms=n_atoms,
        hidden_size=int(cfg.model.hidden_size),
        n_layers=int(cfg.model.n_layers),
        n_heads=int(cfg.model.n_heads),
        dropout_rate=float(cfg.model.dropout_rate),
    )
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"DiT params: {n_params:,}")
    if wandb_run is not None:
        wandb.config.update({"n_params": n_params})

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg.training.lr),
        warmup_steps=int(cfg.training.get("warmup_steps", 5000)),
        decay_steps=int(cfg.training.steps),
        end_value=float(cfg.training.get("end_lr", 1e-5)),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(cfg.training.get("grad_clip_norm", 1.0))),
        optax.adamw(
            schedule,
            weight_decay=float(cfg.training.get("weight_decay", 1e-4)),
            b2=float(cfg.training.get("adam_b2", 0.99)),
        ),
    )
    opt_state = optimizer.init(params)
    train_step = make_train_step(model_config, optimizer, float(cfg.training.beta_min), float(cfg.training.beta_max))
    train = jnp.asarray(train_np)

    for step in range(1, int(cfg.training.steps) + 1):
        key, key_batch, key_step = jax.random.split(key, 3)
        idx = jax.random.randint(key_batch, (int(cfg.training.batch_size),), 0, train.shape[0])
        params, opt_state, loss = train_step(params, opt_state, train[idx], key_step)
        if step == 1 or step % int(cfg.training.log_every) == 0:
            loss_val = float(loss)
            print(f"step={step:06d} loss={loss_val:.6f}")
            if wandb_run is not None:
                wandb.log({"loss": loss_val}, step=step)
        if step % int(cfg.training.eval_every) == 0 or step == int(cfg.training.steps):
            ref_np = train_np[: int(cfg.eval.n_samples)] * std + mean
            sample_sets = []
            sample_paths = []
            for n_sample_steps in eval_step_counts(int(cfg.eval.sample_steps)):
                key, key_sample = jax.random.split(key)
                samples = sample(params, model_config, int(cfg.eval.n_samples), n_sample_steps, key_sample, float(cfg.training.beta_min), float(cfg.training.beta_max))
                samples_np = np.asarray(samples) * std + mean
                samples_path = out_dir / f"latest_samples_steps_{n_sample_steps:03d}.npz"
                np.savez(samples_path, positions=samples_np.astype(np.float32), mean=mean, std=std, sample_steps=n_sample_steps)
                sample_sets.append((n_sample_steps, samples_np))
                sample_paths.append(samples_path)
                if n_sample_steps == int(cfg.eval.sample_steps):
                    np.savez(out_dir / "latest_samples.npz", positions=samples_np.astype(np.float32), mean=mean, std=std, sample_steps=n_sample_steps)
            plot_path = out_dir / f"ramachandran_step_{step:06d}.png"
            plot_ramachandran(ref_np, sample_sets, sequence, plot_path)
            tica_eval = compute_tica_pmf_mjs(
                data_path,
                ref_np,
                sample_sets,
                sequence,
                n_bins=int(cfg.eval.get("tica_bins", 100)),
            )
            tica_plot_path = out_dir / f"tica_step_{step:06d}.png"
            plot_tica_projections(tica_eval, tica_plot_path)
            torus_eval = compute_torus_wasserstein(ref_np, sample_sets, sequence)
            torus_plot_path = out_dir / f"torus_w2_step_{step:06d}.png"
            plot_torus_wasserstein(torus_eval, torus_plot_path)
            for n_sample_steps, metrics in tica_eval["metrics"].items():
                print(
                    f"TICA steps={n_sample_steps} pmf={metrics['pmf']:.6f} "
                    f"mjs={metrics['mjs']:.6f} js={metrics['js']:.6f}"
                )
            for n_sample_steps, metrics in torus_eval["metrics"].items():
                print(f"torus W2 steps={n_sample_steps} w2={metrics['w2']:.6f}")
            print(f"saved {plot_path}")
            if wandb_run is not None:
                eval_log = {
                    "ramachandran/all_sample_steps": wandb.Image(str(plot_path)),
                    "tica/all_sample_steps": wandb.Image(str(tica_plot_path)),
                    "torus_w2/by_sample_steps": wandb.Image(str(torus_plot_path)),
                }
                for n_sample_steps, metrics in tica_eval["metrics"].items():
                    eval_log[f"tica/pmf_steps_{n_sample_steps}"] = metrics["pmf"]
                    eval_log[f"tica/mjs_steps_{n_sample_steps}"] = metrics["mjs"]
                    eval_log[f"tica/js_steps_{n_sample_steps}"] = metrics["js"]
                for n_sample_steps, metrics in torus_eval["metrics"].items():
                    eval_log[f"torus_w2/steps_{n_sample_steps}"] = metrics["w2"]
                wandb.log(eval_log, step=step)
                artifact = wandb.Artifact(f"{cfg.run_name}-eval-{step:06d}", type="evaluation")
                artifact.add_file(str(plot_path))
                artifact.add_file(str(tica_plot_path))
                artifact.add_file(str(torus_plot_path))
                for samples_path in sample_paths:
                    artifact.add_file(str(samples_path))
                wandb.log_artifact(artifact)
        if step % int(cfg.training.checkpoint_every) == 0 or step == int(cfg.training.steps):
            ckpt_path = out_dir / "checkpoints" / f"step_{step:06d}.pkl"
            save_checkpoint(ckpt_path, params, model_config, opt_state, mean, std, sequence, bool(cfg.center_molecule), step, key, float(cfg.training.beta_min), float(cfg.training.beta_max))
            save_checkpoint(out_dir / "checkpoints" / "latest.pkl", params, model_config, opt_state, mean, std, sequence, bool(cfg.center_molecule), step, key, float(cfg.training.beta_min), float(cfg.training.beta_max))
            print(f"saved checkpoint {ckpt_path}")

    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump({"params": params, "config": model_config, "mean": mean, "std": std, "sequence": sequence, "center_molecule": bool(cfg.center_molecule)}, f)
    if wandb_run is not None:
        model_artifact = wandb.Artifact(f"{cfg.run_name}-model", type="model")
        model_artifact.add_file(str(out_dir / "model.pkl"))
        model_artifact.add_file(str(out_dir / "config.json"))
        wandb.log_artifact(model_artifact)
        wandb.finish()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="chignolin_xpred")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
