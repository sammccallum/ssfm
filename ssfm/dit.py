"""Minimal vanilla-JAX DiT for fixed-order coordinate tokens."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def init_attention(key: PRNGKeyArray, hidden_size: int, n_heads: int) -> dict[str, Array]:
    if hidden_size % n_heads != 0:
        raise ValueError("hidden_size must be divisible by n_heads")
    k_qkv, k_out = jax.random.split(key)
    lim = 1 / jnp.sqrt(hidden_size)
    return {
        "qkv_w": jax.random.uniform(k_qkv, (3 * hidden_size, hidden_size), minval=-lim, maxval=lim),
        "out_w": jax.random.uniform(k_out, (hidden_size, hidden_size), minval=-lim, maxval=lim),
        "out_b": jnp.zeros((hidden_size,)),
    }


def apply_attention(
    params: dict[str, Array],
    x: Array,
    *,
    n_heads: int,
    dropout_rate: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> Array:
    n, hidden_size = x.shape
    head_dim = hidden_size // n_heads
    qkv = (x @ params["qkv_w"].T).reshape(n, 3, n_heads, head_dim)
    q, k, v = (jnp.transpose(qkv[:, i], (1, 0, 2)) for i in range(3))
    attn = jax.nn.softmax(jnp.einsum("hnd,hmd->hnm", q, k) * (head_dim**-0.5), axis=-1)
    if dropout_rate and key is not None:
        key, subkey = jax.random.split(key)
        keep = 1.0 - dropout_rate
        attn = jnp.where(jax.random.bernoulli(subkey, keep, attn.shape), attn / keep, 0)
    x = jnp.einsum("hnm,hmd->hnd", attn, v)
    x = jnp.transpose(x, (1, 0, 2)).reshape(n, hidden_size)
    x = x @ params["out_w"].T + params["out_b"]
    if dropout_rate and key is not None:
        _, subkey = jax.random.split(key)
        keep = 1.0 - dropout_rate
        x = jnp.where(jax.random.bernoulli(subkey, keep, x.shape), x / keep, 0)
    return x


def init_mlp(key: PRNGKeyArray, hidden_size: int, mlp_ratio: int = 4) -> dict[str, Array]:
    k1, k2 = jax.random.split(key)
    mlp_size = mlp_ratio * hidden_size
    return {
        "w1": jax.random.uniform(k1, (mlp_size, hidden_size), minval=-(hidden_size**-0.5), maxval=hidden_size**-0.5),
        "b1": jnp.zeros((mlp_size,)),
        "w2": jax.random.uniform(k2, (hidden_size, mlp_size), minval=-(mlp_size**-0.5), maxval=mlp_size**-0.5),
        "b2": jnp.zeros((hidden_size,)),
    }


def apply_mlp(
    params: dict[str, Array],
    x: Array,
    *,
    dropout_rate: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> Array:
    x = jax.nn.gelu(x @ params["w1"].T + params["b1"], approximate=True)
    x = x @ params["w2"].T + params["b2"]
    if dropout_rate and key is not None:
        keep = 1.0 - dropout_rate
        x = jnp.where(jax.random.bernoulli(key, keep, x.shape), x / keep, 0)
    return x


def init_block(
    key: PRNGKeyArray,
    hidden_size: int,
    n_heads: int,
    time_dim: int,
    mlp_ratio: int = 4,
) -> dict[str, Any]:
    k_attn, k_mlp, k_mod = jax.random.split(key, 3)
    return {
        "norm1_scale": jnp.ones((hidden_size,)),
        "norm1_bias": jnp.zeros((hidden_size,)),
        "attn": init_attention(k_attn, hidden_size, n_heads),
        "norm2_scale": jnp.ones((hidden_size,)),
        "norm2_bias": jnp.zeros((hidden_size,)),
        "mlp": init_mlp(k_mlp, hidden_size, mlp_ratio),
        "mod_w": jnp.zeros((6 * hidden_size, time_dim)),
        "mod_b": jnp.zeros((6 * hidden_size,)),
    }


def apply_block(
    params: dict[str, Any],
    x: Array,
    time_emb: Array,
    *,
    n_heads: int,
    dropout_rate: float = 0.0,
    key: PRNGKeyArray | None = None,
) -> Array:
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(
        time_emb @ params["mod_w"].T + params["mod_b"], 6
    )
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    h = (x - mean) * jax.lax.rsqrt(var + 1e-5) * params["norm1_scale"] + params["norm1_bias"]
    h = h * (1 + scale_a) + shift_a
    key, subkey = (jax.random.split(key) if key is not None else (None, None))
    x = x + gate_a * apply_attention(params["attn"], h, n_heads=n_heads, dropout_rate=dropout_rate, key=subkey)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    h = (x - mean) * jax.lax.rsqrt(var + 1e-5) * params["norm2_scale"] + params["norm2_bias"]
    h = h * (1 + scale_m) + shift_m
    key, subkey = (jax.random.split(key) if key is not None else (None, None))
    return x + gate_m * apply_mlp(params["mlp"], h, dropout_rate=dropout_rate, key=subkey)


def init_dit(
    key: PRNGKeyArray,
    *,
    n_atoms: int,
    hidden_size: int = 256,
    n_layers: int = 8,
    n_heads: int = 8,
    mlp_ratio: int = 4,
    time_dim: int | None = None,
    time_fourier_dim: int = 256,
    coord_dim: int = 3,
    output_dim: int | None = None,
    two_time_conditioning: bool = False,
    dropout_rate: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if two_time_conditioning and n_layers % 2:
        raise ValueError("two_time_conditioning requires an even number of layers")
    time_dim = hidden_size if time_dim is None else time_dim
    output_dim = coord_dim if output_dim is None else output_dim
    k_coord, k_pos, k_t1, k_t2, k_final, *k_blocks = jax.random.split(key, 5 + n_layers)
    return {
        "coord_w": jax.random.uniform(k_coord, (hidden_size, coord_dim), minval=-(coord_dim**-0.5), maxval=coord_dim**-0.5),
        "coord_b": jnp.zeros((hidden_size,)),
        "pos_embed": 0.02 * jax.random.normal(k_pos, (n_atoms, hidden_size)),
        "time_w1": jax.random.uniform(k_t1, (time_dim, time_fourier_dim), minval=-(time_fourier_dim**-0.5), maxval=time_fourier_dim**-0.5),
        "time_b1": jnp.zeros((time_dim,)),
        "time_w2": jax.random.uniform(k_t2, (time_dim, time_dim), minval=-(time_dim**-0.5), maxval=time_dim**-0.5),
        "time_b2": jnp.zeros((time_dim,)),
        "blocks": [init_block(k, hidden_size, n_heads, time_dim, mlp_ratio) for k in k_blocks],
        "final_norm_scale": jnp.ones((hidden_size,)),
        "final_norm_bias": jnp.zeros((hidden_size,)),
        "final_mod_w": jnp.zeros((2 * hidden_size, time_dim)),
        "final_mod_b": jnp.zeros((2 * hidden_size,)),
        "final_w": jnp.zeros((output_dim, hidden_size)),
        "final_b": jnp.zeros((output_dim,)),
    }, {
        "n_atoms": n_atoms,
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "time_fourier_dim": time_fourier_dim,
        "coord_dim": coord_dim,
        "output_dim": output_dim,
        "two_time_conditioning": two_time_conditioning,
        "dropout_rate": dropout_rate,
    }


def time_embedding(params: dict[str, Any], config: dict[str, Any], time: Array) -> Array:
    half = config["time_fourier_dim"] // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    emb = jnp.concatenate([jnp.cos(time * freqs), jnp.sin(time * freqs)])
    if config["time_fourier_dim"] % 2:
        emb = jnp.concatenate([emb, jnp.zeros((1,), dtype=emb.dtype)])
    emb = jax.nn.silu(emb @ params["time_w1"].T + params["time_b1"])
    return emb @ params["time_w2"].T + params["time_b2"]


def apply_dit(
    params: dict[str, Any],
    config: dict[str, Any],
    coords: Array,
    time: Array,
    *,
    end_time: Array | None = None,
    key: PRNGKeyArray | None = None,
) -> Array:
    if coords.shape != (config["n_atoms"], config["coord_dim"]):
        raise ValueError(f"coords must have shape {(config['n_atoms'], config['coord_dim'])}, got {coords.shape}")
    x = coords @ params["coord_w"].T + params["coord_b"] + params["pos_embed"]
    time_emb = time_embedding(params, config, time)
    if end_time is not None:
        if not config.get("two_time_conditioning", False):
            raise ValueError("end_time was provided, but this DiT was not initialized with two_time_conditioning=True")
        if len(params["blocks"]) % 2:
            raise ValueError("two-time blockwise alternation requires an even number of blocks")
        end_time_emb = time_embedding(params, config, end_time)
    else:
        end_time_emb = time_emb
    for i, block in enumerate(params["blocks"]):
        key, subkey = (jax.random.split(key) if key is not None else (None, None))
        block_time_emb = time_emb if i % 2 == 0 else end_time_emb
        x = apply_block(
            block,
            x,
            block_time_emb,
            n_heads=config["n_heads"],
            dropout_rate=config["dropout_rate"],
            key=subkey,
        )
    final_time_emb = end_time_emb if end_time is not None else time_emb
    shift, scale = jnp.split(final_time_emb @ params["final_mod_w"].T + params["final_mod_b"], 2)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x = (x - mean) * jax.lax.rsqrt(var + 1e-5) * params["final_norm_scale"] + params["final_norm_bias"]
    return (x * (1 + scale) + shift) @ params["final_w"].T + params["final_b"]
