import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ssfm.flow_map import AbstractEMStepModel
from ssfm.typing import Y


def _downsample(x):
    c, h, w = x.shape
    x = x.reshape(c, h // 2, 2, w // 2, 2)
    return x.mean(axis=(2, 4))


def _upsample(x):
    x = jnp.repeat(x, 2, axis=1)
    x = jnp.repeat(x, 2, axis=2)
    return x


class RMSGroupNorm(eqx.Module):
    num_groups: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, num_groups, eps=1e-4):
        self.num_groups = num_groups
        self.eps = eps

    def __call__(self, x):
        c, h, w = x.shape
        g = self.num_groups
        x = x.reshape(g, c // g, h, w)
        rms = jnp.sqrt(jnp.mean(x**2, axis=(1, 2, 3), keepdims=True)) + self.eps
        x = x / rms
        return x.reshape(c, h, w)


def pixel_norm(x, eps=1e-4):
    return x / (jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True)) + eps)


class FourierTimeEmbedding(eqx.Module):
    freqs: jax.Array
    phases: jax.Array
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, fourier_dim, time_dim, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.freqs = jax.random.normal(k1, (fourier_dim,))
        self.phases = jax.random.uniform(k2, (fourier_dim,))
        self.linear1 = eqx.nn.Linear(fourier_dim, time_dim, use_bias=False, key=k3)
        self.linear2 = eqx.nn.Linear(time_dim, time_dim, use_bias=False, key=k4)

    def __call__(self, time):
        freqs = jax.lax.stop_gradient(self.freqs)
        phases = jax.lax.stop_gradient(self.phases)
        x = jnp.cos(2 * jnp.pi * (freqs * time + phases))
        x = self.linear1(x)
        x = jax.nn.silu(x)
        x = self.linear2(x)
        return x


class ResBlock(eqx.Module):
    norm1: RMSGroupNorm
    conv1: eqx.nn.Conv2d
    time_proj: eqx.nn.Linear
    norm2: RMSGroupNorm
    conv2: eqx.nn.Conv2d
    skip_conv: eqx.nn.Conv2d | None
    resample: str
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_ch,
        out_ch,
        time_dim,
        num_groups=8,
        resample="none",
        dropout_rate=0.0,
        *,
        key,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = RMSGroupNorm(min(num_groups, in_ch))
        self.conv1 = eqx.nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1, use_bias=False, key=k1
        )
        self.time_proj = eqx.nn.Linear(time_dim, out_ch, use_bias=False, key=k2)
        self.norm2 = RMSGroupNorm(min(num_groups, out_ch))
        self.conv2 = eqx.nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, use_bias=False, key=k3
        )
        self.skip_conv = (
            eqx.nn.Conv2d(in_ch, out_ch, kernel_size=1, use_bias=False, key=k4)
            if in_ch != out_ch
            else None
        )
        self.resample = resample
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x, t_emb, *, key=None):
        h = self.norm1(x)
        h = jax.nn.silu(h)
        if self.resample == "down":
            h = _downsample(h)
        elif self.resample == "up":
            h = _upsample(h)
        h = self.conv1(h)

        scale = self.time_proj(t_emb)
        h = self.norm2(h)
        h = h * (1 + scale[:, None, None])
        h = jax.nn.silu(h)
        h = self.dropout(h, key=key)
        h = self.conv2(h)

        skip = x
        if self.resample == "down":
            skip = _downsample(skip)
        elif self.resample == "up":
            skip = _upsample(skip)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        return skip + h


class CosineAttention(eqx.Module):
    qkv_proj: eqx.nn.Conv2d
    out_proj: eqx.nn.Conv2d
    num_heads: int = eqx.field(static=True)

    def __init__(self, channels, num_heads, *, key):
        k1, k2 = jax.random.split(key)
        self.qkv_proj = eqx.nn.Conv2d(
            channels, 3 * channels, kernel_size=1, use_bias=False, key=k1
        )
        self.out_proj = eqx.nn.Conv2d(
            channels, channels, kernel_size=1, use_bias=False, key=k2
        )
        self.num_heads = num_heads

    def __call__(self, x):
        c, h, w = x.shape
        head_dim = c // self.num_heads
        seq_len = h * w

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(3, self.num_heads, head_dim, seq_len)
        qkv = jnp.transpose(qkv, (0, 1, 3, 2))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = pixel_norm(q)
        k = pixel_norm(k)

        attn = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(head_dim)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)

        out = jnp.transpose(out, (0, 2, 1))
        out = out.reshape(c, h, w)
        out = self.out_proj(out)

        return x + out


class EDM2_UNet(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    resolution: int = eqx.field(static=True)
    input_res: int = eqx.field(static=True)
    pad: int = eqx.field(static=True)
    s_embed: FourierTimeEmbedding
    t_embed: FourierTimeEmbedding
    time_fuse1: eqx.nn.Linear
    time_fuse2: eqx.nn.Linear
    init_conv: eqx.nn.Conv2d
    enc_blocks: list
    enc_attns: list
    mid_block: ResBlock
    mid_attn: CosineAttention | None
    dec_blocks: list
    dec_attns: list
    final_norm: RMSGroupNorm
    final_conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        resolution=28,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_groups=8,
        dropout_rate=0.0,
        attn_resolutions=None,
        head_dim=64,
        num_res_blocks=4,
        *,
        key,
    ):
        n_keys = 3 * num_res_blocks * len(channel_mult) + 20
        keys = jax.random.split(key, n_keys)
        ki = iter(keys)

        self.in_channels = in_channels
        time_dim = 4 * base_channels
        channels = [base_channels * m for m in channel_mult]

        self.s_embed = FourierTimeEmbedding(base_channels, time_dim, key=next(ki))
        self.t_embed = FourierTimeEmbedding(base_channels, time_dim, key=next(ki))
        self.time_fuse1 = eqx.nn.Linear(
            2 * time_dim, time_dim, use_bias=False, key=next(ki)
        )
        self.time_fuse2 = eqx.nn.Linear(
            time_dim, time_dim, use_bias=False, key=next(ki)
        )
        self.init_conv = eqx.nn.Conv2d(
            in_channels + 1,
            channels[0],
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=next(ki),
        )

        input_res = 2 ** math.ceil(math.log2(max(resolution, 2)))
        pad = (input_res - resolution) // 2
        self.resolution = resolution
        self.out_channels = out_channels
        self.input_res = input_res
        self.pad = pad

        if attn_resolutions is None:
            attn_set = {input_res // (2**i) for i in range(len(channels))}
        else:
            attn_set = set(attn_resolutions)

        enc_blocks = []
        enc_attns = []
        prev_ch = channels[0]
        res = input_res
        for i, ch_out in enumerate(channels):
            resample = "down" if i > 0 else "none"
            if i > 0:
                res = res // 2
            level_blocks = []
            level_attns = []
            for j in range(num_res_blocks):
                in_ch = prev_ch if j == 0 else ch_out
                rs = resample if j == 0 else "none"
                level_blocks.append(
                    ResBlock(
                        in_ch,
                        ch_out,
                        time_dim,
                        num_groups,
                        resample=rs,
                        dropout_rate=dropout_rate,
                        key=next(ki),
                    )
                )
                level_attns.append(
                    CosineAttention(ch_out, ch_out // head_dim, key=next(ki))
                    if res in attn_set
                    else None
                )
            enc_blocks.append(level_blocks)
            enc_attns.append(level_attns)
            prev_ch = ch_out
        self.enc_blocks = enc_blocks
        self.enc_attns = enc_attns

        mid_res = input_res // (2 ** (len(channels) - 1))
        self.mid_block = ResBlock(
            channels[-1],
            channels[-1],
            time_dim,
            num_groups,
            dropout_rate=dropout_rate,
            key=next(ki),
        )
        self.mid_attn = (
            CosineAttention(channels[-1], channels[-1] // head_dim, key=next(ki))
            if mid_res in attn_set
            else None
        )

        dec_blocks = []
        dec_attns = []
        rev_channels = list(reversed(channels))
        dec_res = mid_res
        for i in range(len(rev_channels)):
            ch_in = rev_channels[i]
            ch_out = (
                rev_channels[i + 1] if i + 1 < len(rev_channels) else rev_channels[i]
            )
            has_up = i < len(rev_channels) - 1
            resample_last = "up" if has_up else "none"
            out_res = dec_res * 2 if has_up else dec_res
            level_blocks = []
            level_attns = []
            for j in range(num_res_blocks):
                if j == 0:
                    in_ch, out_ch, rs = 2 * ch_in, ch_in, "none"
                elif j == num_res_blocks - 1:
                    in_ch, out_ch, rs = ch_in, ch_out, resample_last
                else:
                    in_ch, out_ch, rs = ch_in, ch_in, "none"
                level_blocks.append(
                    ResBlock(
                        in_ch,
                        out_ch,
                        time_dim,
                        num_groups,
                        resample=rs,
                        dropout_rate=dropout_rate,
                        key=next(ki),
                    )
                )
                attn_ch = out_ch
                attn_res = out_res if j == num_res_blocks - 1 else dec_res
                level_attns.append(
                    CosineAttention(attn_ch, attn_ch // head_dim, key=next(ki))
                    if attn_res in attn_set
                    else None
                )
            dec_blocks.append(level_blocks)
            dec_attns.append(level_attns)
            dec_res = out_res
        self.dec_blocks = dec_blocks
        self.dec_attns = dec_attns

        self.final_norm = RMSGroupNorm(min(num_groups, channels[0]))
        self.final_conv = eqx.nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=next(ki),
        )

    def __call__(self, x, s, t, *, key=None):
        x = x.reshape(self.in_channels, self.resolution, self.resolution)
        if self.pad > 0:
            x = jnp.pad(x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)))
        x = jnp.concatenate([x, jnp.ones((1, self.input_res, self.input_res))], axis=0)

        s_emb = self.s_embed(s)
        t_emb = self.t_embed(t)
        t_emb = self.time_fuse2(
            jax.nn.silu(self.time_fuse1(jnp.concatenate([s_emb, t_emb])))
        )
        h = self.init_conv(x)

        def _maybe_split_key(key):
            if key is None:
                return None, None
            return jax.random.split(key)

        skips = []
        for level_blocks, level_attns in zip(self.enc_blocks, self.enc_attns):
            for block, attn in zip(level_blocks, level_attns):
                key, subkey = _maybe_split_key(key)
                h = block(h, t_emb, key=subkey)
                if attn is not None:
                    h = attn(h)
            skips.append(h)

        key, subkey = _maybe_split_key(key)
        h = self.mid_block(h, t_emb, key=subkey)
        if self.mid_attn is not None:
            h = self.mid_attn(h)

        for level_blocks, level_attns in zip(self.dec_blocks, self.dec_attns):
            h = jnp.concatenate([h, skips.pop()], axis=0)
            for block, attn in zip(level_blocks, level_attns):
                key, subkey = _maybe_split_key(key)
                h = block(h, t_emb, key=subkey)
                if attn is not None:
                    h = attn(h)

        h = self.final_norm(h)
        h = jax.nn.silu(h)
        h = self.final_conv(h)
        if self.pad > 0:
            h = h[
                :,
                self.pad : self.pad + self.resolution,
                self.pad : self.pad + self.resolution,
            ]
        return h


class EDM2EMStepModel(AbstractEMStepModel):
    drift_net: EDM2_UNet
    diffusion_net: EDM2_UNet

    def drift(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key=None,
    ) -> Y:
        x = jnp.concatenate([ys, W, H, K], axis=0)
        return self.drift_net(x, s, t, key=key)

    def diffusion(
        self,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key=None,
    ) -> Y:
        x = jnp.concatenate([W, H, K], axis=0)
        return self.diffusion_net(x, s, t, key=key)
