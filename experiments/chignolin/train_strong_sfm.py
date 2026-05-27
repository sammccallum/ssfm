"""Train a strong stochastic flow map on Chignolin coordinates.

This keeps the same dataset, logging, evaluation, and checkpoint behavior as
train_xpred_diffusion.py, but trains an interval-conditioned strong stochastic
flow map with learned drift and diffusion components.
"""

from __future__ import annotations

import json
import os
import pickle
import gc
from pathlib import Path
from typing import Any


os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import diffrax
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


SSFM_BROWNIAN_COEFFS = 3
DATA_COORD_DIM = 3
DRIFT_CONTEXT_DIM = DATA_COORD_DIM * (1 + SSFM_BROWNIAN_COEFFS)
DIFFUSION_CONTEXT_DIM = DATA_COORD_DIM * SSFM_BROWNIAN_COEFFS


def vp_beta(s: jnp.ndarray, beta_min: float, beta_max: float) -> jnp.ndarray:
    return beta_min + s * (beta_max - beta_min)


def drift_features(
    x_s: jnp.ndarray,
    brownian_increment: jnp.ndarray,
    brownian_h: jnp.ndarray,
    brownian_k: jnp.ndarray,
    s: jnp.ndarray,
    t: jnp.ndarray,
    pass_brownian_coefficients: bool,
) -> jnp.ndarray:
    if not pass_brownian_coefficients:
        return jnp.concatenate(
            [x_s, jnp.zeros_like(brownian_increment), jnp.zeros_like(brownian_h), jnp.zeros_like(brownian_k)],
            axis=-1,
        )
    return jnp.concatenate(
        [
            x_s,
            brownian_increment,
            brownian_h,
            brownian_k,
        ],
        axis=-1,
    )


def diffusion_features(
    brownian_increment: jnp.ndarray,
    brownian_h: jnp.ndarray,
    brownian_k: jnp.ndarray,
    s: jnp.ndarray,
    t: jnp.ndarray,
    pass_brownian_coefficients: bool,
) -> jnp.ndarray:
    if not pass_brownian_coefficients:
        return jnp.concatenate(
            [jnp.zeros_like(brownian_increment), jnp.zeros_like(brownian_h), jnp.zeros_like(brownian_k)],
            axis=-1,
        )
    return jnp.concatenate(
        [
            brownian_increment,
            brownian_h,
            brownian_k,
        ],
        axis=-1,
    )


def model_times(
    s: jnp.ndarray,
    t: jnp.ndarray,
    pass_target_time: bool,
    pass_any_time: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if not pass_any_time:
        fixed_time = jnp.zeros_like(s)
        return fixed_time, fixed_time
    return s, t if pass_target_time else s


def init_drift(key: jnp.ndarray, n_atoms: int, model_cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    params, config = init_dit(
        key,
        n_atoms=n_atoms,
        hidden_size=int(model_cfg.hidden_size),
        n_layers=int(model_cfg.n_layers),
        n_heads=int(model_cfg.n_heads),
        coord_dim=DRIFT_CONTEXT_DIM,
        output_dim=DATA_COORD_DIM,
        two_time_conditioning=True,
        dropout_rate=float(model_cfg.dropout_rate),
    )
    config["data_coord_dim"] = DATA_COORD_DIM
    config["brownian_coeffs"] = SSFM_BROWNIAN_COEFFS
    return params, config


def init_diffusion(key: jnp.ndarray, n_atoms: int, model_cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    params, config = init_dit(
        key,
        n_atoms=n_atoms,
        hidden_size=int(model_cfg.hidden_size),
        n_layers=int(model_cfg.get("diffusion_n_layers", 2)),
        n_heads=int(model_cfg.n_heads),
        coord_dim=DIFFUSION_CONTEXT_DIM,
        output_dim=DATA_COORD_DIM,
        two_time_conditioning=True,
        dropout_rate=float(model_cfg.dropout_rate),
    )
    config["data_coord_dim"] = DATA_COORD_DIM
    config["brownian_coeffs"] = SSFM_BROWNIAN_COEFFS
    return params, config


def apply_drift(
    params: dict[str, Any],
    config: dict[str, Any],
    x_s: jnp.ndarray,
    s: jnp.ndarray,
    t: jnp.ndarray,
    brownian_increment: jnp.ndarray,
    brownian_h: jnp.ndarray,
    brownian_k: jnp.ndarray,
    pass_brownian_coefficients: bool,
    pass_target_time: bool,
    pass_any_time: bool,
    key: jnp.ndarray,
) -> jnp.ndarray:
    model_input = drift_features(x_s, brownian_increment, brownian_h, brownian_k, s, t, pass_brownian_coefficients)
    model_time, model_end_time = model_times(s, t, pass_target_time, pass_any_time)
    return apply_dit(params, config, model_input, model_time, end_time=model_end_time, key=key)


def apply_diffusion(
    params: dict[str, Any],
    config: dict[str, Any],
    s: jnp.ndarray,
    t: jnp.ndarray,
    brownian_increment: jnp.ndarray,
    brownian_h: jnp.ndarray,
    brownian_k: jnp.ndarray,
    pass_brownian_coefficients: bool,
    pass_target_time: bool,
    pass_any_time: bool,
    key: jnp.ndarray,
) -> jnp.ndarray:
    model_input = diffusion_features(brownian_increment, brownian_h, brownian_k, s, t, pass_brownian_coefficients)
    model_time, model_end_time = model_times(s, t, pass_target_time, pass_any_time)
    return apply_dit(params, config, model_input, model_time, end_time=model_end_time, key=key)


def apply_strong_sfm(
    params: dict[str, Any],
    config: dict[str, Any],
    x_s: jnp.ndarray,
    s: jnp.ndarray,
    t: jnp.ndarray,
    brownian_increment: jnp.ndarray,
    brownian_h: jnp.ndarray,
    brownian_k: jnp.ndarray,
    pass_brownian_coefficients: bool,
    pass_target_time: bool,
    pass_any_time: bool,
    key: jnp.ndarray | None,
) -> jnp.ndarray:
    if key is None:
        key_drift = key_diffusion = None
    else:
        key_drift, key_diffusion = jax.random.split(key)
    drift = apply_drift(
        params["drift"], config["drift"], x_s, s, t, brownian_increment, brownian_h, brownian_k,
        pass_brownian_coefficients, pass_target_time, pass_any_time, key_drift,
    )
    diffusion = apply_diffusion(
        params["diffusion"], config["diffusion"], s, t, brownian_increment, brownian_h, brownian_k,
        pass_brownian_coefficients, pass_target_time, pass_any_time, key_diffusion,
    )
    return x_s + (t - s) * drift + brownian_increment * diffusion


def init_uncertainty(key: jnp.ndarray, fourier_dim: int) -> dict[str, jnp.ndarray]:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return {
        "freqs_s": jax.random.normal(k1, (fourier_dim,)),
        "phases_s": jax.random.uniform(k2, (fourier_dim,)),
        "freqs_t": jax.random.normal(k3, (fourier_dim,)),
        "phases_t": jax.random.uniform(k4, (fourier_dim,)),
        "linear_w": jax.random.normal(k5, (1, 2 * fourier_dim)) / jnp.sqrt(2.0 * fourier_dim),
        "linear_b": jnp.zeros((1,)),
    }


def apply_uncertainty(params: dict[str, jnp.ndarray], s: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    feat_s = jnp.cos(
        2.0 * jnp.pi * (jax.lax.stop_gradient(params["freqs_s"]) * s + jax.lax.stop_gradient(params["phases_s"]))
    )
    feat_t = jnp.cos(
        2.0 * jnp.pi * (jax.lax.stop_gradient(params["freqs_t"]) * t + jax.lax.stop_gradient(params["phases_t"]))
    )
    features = jnp.concatenate([feat_s, feat_t])
    return (features @ params["linear_w"].T + params["linear_b"]).squeeze()


def ema_update(params: dict[str, Any], ema_params: dict[str, Any], decay: float) -> dict[str, Any]:
    return jax.tree.map(lambda ema, p: decay * ema + (1.0 - decay) * p, ema_params, params)


def make_train_step(
    model_config: dict[str, Any],
    optimizer: optax.GradientTransformation,
    beta_min: float,
    beta_max: float,
    reverse_eta: float,
    eta: float,
    dt: float,
    h_max: float,
    t_eps: float,
    ema_decay: float,
    total_steps: int,
    horizon_curriculum: bool,
    horizon_curriculum_fraction: float,
    horizon_curriculum_power: float,
    log_uniform_h: bool,
    split_batch: bool,
    pass_brownian_coefficients: bool,
    pass_target_time: bool,
    pass_any_time: bool,
):
    @jax.jit
    def train_step(
        params: dict[str, Any],
        ema_params: dict[str, Any],
        opt_state: optax.OptState,
        batch: jnp.ndarray,
        key: jnp.ndarray,
        step: jnp.ndarray,
    ):
        if horizon_curriculum:
            curriculum_steps = jnp.maximum(1.0, horizon_curriculum_fraction * float(total_steps))
            progress = jnp.clip((step - 1.0) / curriculum_steps, 0.0, 1.0)
            horizon_progress = progress ** horizon_curriculum_power
            h_max_step = dt * (h_max / dt) ** horizon_progress
        else:
            horizon_progress = jnp.asarray(1.0)
            h_max_step = jnp.asarray(h_max)
        eta_step = jnp.asarray(eta)
        n_local = max(1, min(batch.shape[0] - 1, int(batch.shape[0] * eta)))

        def sample_h(key_h: jnp.ndarray, shape: tuple[int, ...], lower: float, upper: jnp.ndarray) -> jnp.ndarray:
            lower_arr = jnp.asarray(lower)
            upper_arr = jnp.maximum(jnp.asarray(upper), lower_arr)
            if log_uniform_h:
                log_h = jax.random.uniform(key_h, shape, minval=jnp.log(lower_arr), maxval=jnp.log(upper_arr))
                return jnp.exp(log_h)
            return jax.random.uniform(key_h, shape, minval=lower_arr, maxval=upper_arr)

        def local_loss_fn(p: dict[str, Any], local_batch: jnp.ndarray, local_key: jnp.ndarray) -> jnp.ndarray:
            key_h, key_s, key_noise, key_vbt, key_model = jax.random.split(local_key, 5)
            batch_size = local_batch.shape[0]
            h = sample_h(key_h, (batch_size,), dt / 2.0, jnp.asarray(dt))
            s = jax.random.uniform(key_s, (batch_size,), minval=t_eps + h, maxval=1.0)
            t = jnp.clip(s - h, t_eps, 1.0)
            noise = jax.random.normal(key_noise, local_batch.shape)
            alpha_s, sigma_s = vp_alpha_sigma(s[:, None, None], beta_min, beta_max)
            x_s = alpha_s * local_batch + sigma_s * noise
            vbt_keys = jax.random.split(key_vbt, batch_size)
            model_keys = jax.random.split(key_model, batch_size)

            def one(xi, x0, si, ti, hi, vk, mk):
                vbt = diffrax.VirtualBrownianTree(
                    t0=t_eps,
                    t1=1.0,
                    tol=hi / 4.0,
                    shape=xi.shape,
                    key=vk,
                    levy_area=diffrax.SpaceTimeTimeLevyArea,
                )
                levy = vbt.evaluate(si, ti, use_levy=True)
                beta_s = vp_beta(si, beta_min, beta_max)
                alpha_s_i, sigma_s_i = vp_alpha_sigma(si, beta_min, beta_max)
                lambda_s = 0.5 * (1.0 + reverse_eta**2) * beta_s / (sigma_s_i**2)
                target_drift = (-0.5 * beta_s + lambda_s) * xi - lambda_s * alpha_s_i * x0
                target_diffusion = reverse_eta * jnp.sqrt(beta_s) * levy.W
                target = xi + (ti - si) * target_drift + target_diffusion
                pred = apply_strong_sfm(
                    p, model_config, xi, si, ti, levy.W, levy.H, levy.K,
                    pass_brownian_coefficients, pass_target_time, pass_any_time, mk,
                )
                ssfm_local = jnp.mean(jnp.square(pred - jax.lax.stop_gradient(target))) / hi
                u = apply_uncertainty(p["uncertainty"], si, ti)
                return jnp.exp(-u) * ssfm_local + u

            return jnp.mean(jax.vmap(one)(x_s, local_batch, s, t, h, vbt_keys, model_keys))

        def distill_loss_fn(p: dict[str, Any], distill_batch: jnp.ndarray, distill_key: jnp.ndarray) -> jnp.ndarray:
            key_h, key_s, key_noise, key_vbt, key_student, key_teacher = jax.random.split(distill_key, 6)
            batch_size = distill_batch.shape[0]
            h = sample_h(key_h, (batch_size,), dt, jnp.minimum(h_max_step, 1.0 - t_eps))
            s = jax.random.uniform(key_s, (batch_size,), minval=t_eps + h, maxval=1.0)
            t = jnp.clip(s - h, t_eps, 1.0)
            u = jnp.clip(s - 0.5 * h, t_eps, 1.0)
            noise = jax.random.normal(key_noise, distill_batch.shape)
            alpha_s, sigma_s = vp_alpha_sigma(s[:, None, None], beta_min, beta_max)
            x_s = alpha_s * distill_batch + sigma_s * noise
            vbt_keys = jax.random.split(key_vbt, batch_size)
            student_keys = jax.random.split(key_student, batch_size)
            teacher_keys = jax.random.split(key_teacher, (batch_size, 2))

            def one(xi, si, ui, ti, hi, vk, sk, tk):
                vbt = diffrax.VirtualBrownianTree(
                    t0=t_eps,
                    t1=1.0,
                    tol=hi / 4.0,
                    shape=xi.shape,
                    key=vk,
                    levy_area=diffrax.SpaceTimeTimeLevyArea,
                )
                levy_st = vbt.evaluate(si, ti, use_levy=True)
                x_one = apply_strong_sfm(
                    p["params"], model_config, xi, si, ti, levy_st.W, levy_st.H, levy_st.K,
                    pass_brownian_coefficients, pass_target_time, pass_any_time, sk,
                )
                levy_su = vbt.evaluate(si, ui, use_levy=True)
                x_u = apply_strong_sfm(
                    p["ema"], model_config, xi, si, ui, levy_su.W, levy_su.H, levy_su.K,
                    pass_brownian_coefficients, pass_target_time, pass_any_time, None,
                )
                levy_ut = vbt.evaluate(ui, ti, use_levy=True)
                x_two = apply_strong_sfm(
                    p["ema"], model_config, x_u, ui, ti, levy_ut.W, levy_ut.H, levy_ut.K,
                    pass_brownian_coefficients, pass_target_time, pass_any_time, None,
                )
                raw_loss = jnp.mean(jnp.square(x_one - jax.lax.stop_gradient(x_two))) / hi
                uncertainty = apply_uncertainty(p["params"]["uncertainty"], si, ti)
                return jnp.exp(-uncertainty) * raw_loss + uncertainty

            return jnp.mean(jax.vmap(one)(x_s, s, u, t, h, vbt_keys, student_keys, teacher_keys))

        def loss_fn(p: dict[str, Any]):
            key_local, key_distill = jax.random.split(key)
            if split_batch:
                local_batch = batch[:n_local]
                distill_batch = batch[n_local:]
            else:
                local_batch = batch
                distill_batch = batch
            local_loss = local_loss_fn(p["params"], local_batch, key_local)
            distill_loss = distill_loss_fn(p, distill_batch, key_distill)
            loss = eta_step * local_loss + (1.0 - eta_step) * distill_loss
            return loss, (local_loss, distill_loss)

        diff_args = {"params": params, "ema": ema_params}
        (loss, (local_loss, distill_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(diff_args)
        params_grads = grads["params"]
        updates, opt_state = optimizer.update(params_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = ema_update({"drift": params["drift"], "diffusion": params["diffusion"]}, ema_params, ema_decay)
        return params, ema_params, opt_state, loss, local_loss, distill_loss, h_max_step, eta_step, horizon_progress

    return train_step


def sample(
    params: dict[str, Any],
    config: dict[str, Any],
    n_samples: int,
    n_steps: int,
    key: jnp.ndarray,
    t_eps: float,
    pass_brownian_coefficients: bool,
    pass_target_time: bool,
    pass_any_time: bool,
) -> jnp.ndarray:
    key, key_init, key_vbt = jax.random.split(key, 3)
    data_coord_dim = int(config["drift"].get("data_coord_dim", DATA_COORD_DIM))
    x = jax.random.normal(key_init, (n_samples, config["drift"]["n_atoms"], data_coord_dim))
    time_grid = jnp.clip(jnp.linspace(1.0, t_eps, n_steps + 1), t_eps, 1.0)

    def sample_one(x0: jnp.ndarray, k_vbt: jnp.ndarray, k_model: jnp.ndarray) -> jnp.ndarray:
        vbt = diffrax.VirtualBrownianTree(
            t0=t_eps,
            t1=1.0,
            tol=(1.0 - t_eps) / (4.0 * n_steps),
            shape=x0.shape,
            key=k_vbt,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )

        def scan_fn(xi, i):
            s = time_grid[i]
            t = time_grid[i + 1]
            levy = vbt.evaluate(s, t, use_levy=True)
            step_key = jax.random.fold_in(k_model, i)
            return apply_strong_sfm(
                params, config, xi, s, t, levy.W, levy.H, levy.K,
                pass_brownian_coefficients, pass_target_time, pass_any_time, step_key,
            ), None

        x_final, _ = jax.lax.scan(scan_fn, x0, jnp.arange(n_steps))
        return x_final

    vbt_keys = jax.random.split(key_vbt, n_samples)
    model_keys = jax.random.split(key, n_samples)
    return jax.vmap(sample_one)(x, vbt_keys, model_keys)


def save_checkpoint(
    path: Path,
    params: dict[str, Any],
    ema_params: dict[str, Any],
    config: dict[str, Any],
    opt_state: optax.OptState,
    mean: np.ndarray,
    std: np.ndarray,
    sequence: str,
    center_molecule: bool,
    step: int,
    key: jnp.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({
            "params": params,
            "ema_params": ema_params,
            "config": config,
            "opt_state": opt_state,
            "mean": mean,
            "std": std,
            "sequence": sequence,
            "center_molecule": center_molecule,
            "step": step,
            "key": key,
        }, f)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("rb") as f:
        return pickle.load(f)


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
    key, key_drift, key_diffusion, key_uncertainty = jax.random.split(key, 4)
    drift_params, drift_config = init_drift(key_drift, n_atoms, cfg.model)
    diffusion_params, diffusion_config = init_diffusion(key_diffusion, n_atoms, cfg.model)
    uncertainty_params = init_uncertainty(key_uncertainty, int(cfg.model.get("uncertainty_fourier_dim", 128)))
    params = {"drift": drift_params, "diffusion": diffusion_params, "uncertainty": uncertainty_params}
    model_config = {"drift": drift_config, "diffusion": diffusion_config}
    ema_params = {"drift": drift_params, "diffusion": diffusion_params}
    n_drift_params = sum(x.size for x in jax.tree.leaves(drift_params))
    n_diffusion_params = sum(x.size for x in jax.tree.leaves(diffusion_params))
    n_uncertainty_params = sum(x.size for x in jax.tree.leaves(uncertainty_params))
    n_params = n_drift_params + n_diffusion_params + n_uncertainty_params
    print(
        f"DiT params: {n_params:,} drift={n_drift_params:,} "
        f"diffusion={n_diffusion_params:,} uncertainty={n_uncertainty_params:,}"
    )
    if wandb_run is not None:
        wandb.config.update({
            "n_params": n_params,
            "n_drift_params": n_drift_params,
            "n_diffusion_params": n_diffusion_params,
            "n_uncertainty_params": n_uncertainty_params,
        })

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg.training.lr),
        warmup_steps=int(cfg.training.get("warmup_steps", 5000)),
        decay_steps=int(cfg.training.steps),
        end_value=float(cfg.training.get("end_lr", 1e-5)),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(cfg.training.get("grad_clip_norm", 1.0))),
        optax.adamw(schedule, weight_decay=float(cfg.training.get("weight_decay", 1e-4)), b2=float(cfg.training.get("adam_b2", 0.99))),
    )
    opt_state = optimizer.init(params)
    start_step = 0
    resume_from = cfg.training.get("resume_from")
    if resume_from:
        ckpt = load_checkpoint(resume_from)
        ckpt_sequence = ckpt.get("sequence")
        if ckpt_sequence is not None and str(ckpt_sequence) != sequence:
            raise ValueError(f"resume checkpoint sequence {ckpt_sequence!r} does not match data sequence {sequence!r}")
        if bool(ckpt.get("center_molecule", bool(cfg.center_molecule))) != bool(cfg.center_molecule):
            raise ValueError("resume checkpoint center_molecule does not match config")
        params = ckpt["params"]
        ema_params = {
            "drift": ckpt.get("ema_params", ckpt["params"])["drift"],
            "diffusion": ckpt.get("ema_params", ckpt["params"])["diffusion"],
        }
        opt_state = ckpt["opt_state"]
        key = ckpt.get("key", key)
        start_step = int(ckpt.get("step", 0))
        print(f"resumed checkpoint {resume_from} at step={start_step:06d}")
    train_step = make_train_step(
        model_config,
        optimizer,
        float(cfg.training.beta_min),
        float(cfg.training.beta_max),
        float(cfg.training.reverse_eta),
        float(cfg.training.eta),
        float(cfg.training.dt),
        float(cfg.training.h_max),
        float(cfg.training.t_eps),
        float(cfg.training.ema_decay),
        int(cfg.training.steps),
        bool(cfg.training.get("horizon_curriculum", True)),
        float(cfg.training.get("horizon_curriculum_fraction", 0.3)),
        float(cfg.training.get("horizon_curriculum_power", 2.0)),
        bool(cfg.training.get("log_uniform_h", False)),
        bool(cfg.training.get("split_batch", False)),
        bool(cfg.training.get("pass_brownian_coefficients", True)),
        bool(cfg.training.get("pass_target_time", True)),
        bool(cfg.training.get("pass_any_time", True)),
    )
    train = jnp.asarray(train_np)

    for step in range(start_step + 1, int(cfg.training.steps) + 1):
        key, key_batch, key_step = jax.random.split(key, 3)
        idx = jax.random.randint(key_batch, (int(cfg.training.batch_size),), 0, train.shape[0])
        (
            params,
            ema_params,
            opt_state,
            loss,
            local_loss,
            distill_loss,
            h_max_step,
            eta_step,
            horizon_progress,
        ) = train_step(
            params, ema_params, opt_state, train[idx], key_step, jnp.asarray(step, dtype=jnp.float32)
        )
        if step == 1 or step % int(cfg.training.log_every) == 0:
            loss_val = float(loss)
            local_val = float(local_loss)
            distill_val = float(distill_loss)
            h_max_val = float(h_max_step)
            eta_val = float(eta_step)
            progress_val = float(horizon_progress)
            print(
                f"step={step:06d} loss={loss_val:.6f} local={local_val:.6f} "
                f"distill={distill_val:.6f} h_max_curr={h_max_val:.6g} eta={eta_val:.4f}"
            )
            if wandb_run is not None:
                wandb.log(
                    {
                        "loss": loss_val,
                        "loss/local": local_val,
                        "loss/distill": distill_val,
                        "curriculum/h_max": h_max_val,
                        "curriculum/eta": eta_val,
                        "curriculum/progress": progress_val,
                    },
                    step=step,
                )
        if step % int(cfg.training.eval_every) == 0 or step == int(cfg.training.steps):
            ref_np = train_np[: int(cfg.eval.n_samples)] * std + mean
            sample_sets = []
            sample_paths = []
            for n_sample_steps in eval_step_counts(int(cfg.eval.sample_steps)):
                key, key_sample = jax.random.split(key)
                samples = sample(
                {"drift": ema_params["drift"], "diffusion": ema_params["diffusion"], "uncertainty": params["uncertainty"]},
                model_config,
                    int(cfg.eval.n_samples),
                    n_sample_steps,
                    key_sample,
                    float(cfg.training.t_eps),
                    bool(cfg.training.get("pass_brownian_coefficients", True)),
                    bool(cfg.training.get("pass_target_time", True)),
                    bool(cfg.training.get("pass_any_time", True)),
                )
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
            del sample_sets, sample_paths, tica_eval, torus_eval
            gc.collect()
        if step % int(cfg.training.checkpoint_every) == 0 or step == int(cfg.training.steps):
            ckpt_path = out_dir / "checkpoints" / f"step_{step:06d}.pkl"
            ckpt_ema_params = {"drift": ema_params["drift"], "diffusion": ema_params["diffusion"], "uncertainty": params["uncertainty"]}
            save_checkpoint(ckpt_path, params, ckpt_ema_params, model_config, opt_state, mean, std, sequence, bool(cfg.center_molecule), step, key)
            save_checkpoint(out_dir / "checkpoints" / "latest.pkl", params, ckpt_ema_params, model_config, opt_state, mean, std, sequence, bool(cfg.center_molecule), step, key)
            print(f"saved checkpoint {ckpt_path}")

    with (out_dir / "model.pkl").open("wb") as f:
        pickle.dump({
            "params": params,
            "ema_params": {"drift": ema_params["drift"], "diffusion": ema_params["diffusion"], "uncertainty": params["uncertainty"]},
            "config": model_config,
            "mean": mean,
            "std": std,
            "sequence": sequence,
            "center_molecule": bool(cfg.center_molecule),
        }, f)
    if wandb_run is not None:
        model_artifact = wandb.Artifact(f"{cfg.run_name}-model", type="model")
        model_artifact.add_file(str(out_dir / "model.pkl"))
        model_artifact.add_file(str(out_dir / "config.json"))
        wandb.log_artifact(model_artifact)
        wandb.finish()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="chignolin_strong_sfm")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
