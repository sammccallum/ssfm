import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ssfm.diffusion import AbstractDiffusion
from ssfm.flow_map import AbstractFlowMap
from ssfm.typing import BatchY, Y


class UncertaintyMLP(eqx.Module):
    freqs_s: jax.Array
    phases_s: jax.Array
    freqs_t: jax.Array
    phases_t: jax.Array
    linear: eqx.nn.Linear

    def __init__(self, fourier_dim: int, *, key: PRNGKeyArray):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.freqs_s = jax.random.normal(k1, (fourier_dim,))
        self.phases_s = jax.random.uniform(k2, (fourier_dim,))
        self.freqs_t = jax.random.normal(k3, (fourier_dim,))
        self.phases_t = jax.random.uniform(k4, (fourier_dim,))
        self.linear = eqx.nn.Linear(2 * fourier_dim, 1, key=k5)

    def __call__(self, s: Float[Array, ""], t: Float[Array, ""]) -> Float[Array, ""]:
        feat_s = jnp.cos(
            2
            * jnp.pi
            * (
                jax.lax.stop_gradient(self.freqs_s) * s
                + jax.lax.stop_gradient(self.phases_s)
            )
        )
        feat_t = jnp.cos(
            2
            * jnp.pi
            * (
                jax.lax.stop_gradient(self.freqs_t) * t
                + jax.lax.stop_gradient(self.phases_t)
            )
        )
        return self.linear(jnp.concatenate([feat_s, feat_t])).squeeze()


class UncertaintyScoreLoss(eqx.Module):
    diffusion: AbstractDiffusion
    dt: float
    t_eps: float = 1e-5

    def _single(
        self,
        flow_map: AbstractFlowMap,
        y0: Y,
        s: Float[Array, ""],
        h: Float[Array, ""],
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        sample_key, vbt_key, dropout_key = jax.random.split(key, 3)

        ys, score = self.diffusion.forward_sample(s, y0, sample_key)

        s = jnp.clip(s, self.t_eps, 1.0 - self.t_eps)
        t = jnp.clip(s - h, self.t_eps, 1.0)

        drift = self.diffusion.reverse_drift(s, ys, score)
        g = self.diffusion.reverse_diffusion(s)

        vbt = diffrax.VirtualBrownianTree(
            t0=self.t_eps,
            t1=1.0,
            tol=h / 4,
            shape=y0.shape,
            key=vbt_key,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )
        levy = vbt.evaluate(s, t, use_levy=True)

        yt_target = ys + drift * (t - s) + g * levy.W
        yt_pred = flow_map(ys, s, t, levy.W, levy.H, levy.K, key=dropout_key)  # pyright: ignore

        raw_loss = jnp.sum((yt_pred - jax.lax.stop_gradient(yt_target)) ** 2) / h
        u = u_mlp(s, t)
        return jnp.exp(-u) * raw_loss + u

    def value(
        self,
        flow_map: AbstractFlowMap,
        ema_flow_map: AbstractFlowMap,
        y0_batch: BatchY,
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        batch_size = y0_batch.shape[0]
        key_h, key_s, key_vbt = jax.random.split(key, 3)

        h_vals = jax.random.uniform(
            key_h, (batch_size,), minval=self.dt / 2, maxval=self.dt
        )
        s_vals = jax.random.uniform(
            key_s, (batch_size,), minval=self.t_eps + h_vals, maxval=1.0
        )
        vbt_keys = jax.random.split(key_vbt, batch_size)

        losses = eqx.filter_vmap(
            lambda y0, s, h, k: self._single(flow_map, y0, s, h, k, u_mlp)
        )(y0_batch, s_vals, h_vals, vbt_keys)

        return jnp.mean(losses)


class UncertaintyDistillationLoss(eqx.Module):
    diffusion: AbstractDiffusion
    dt: float
    h_max: float
    t_eps: float = 1e-5

    def _single(
        self,
        flow_map: AbstractFlowMap,
        ema_flow_map: AbstractFlowMap,
        y0: Y,
        s: Float[Array, ""],
        h: Float[Array, ""],
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        sample_key, vbt_key, dropout_key = jax.random.split(key, 3)
        ema_flow_map = eqx.nn.inference_mode(ema_flow_map)

        ys, _ = self.diffusion.forward_sample(s, y0, sample_key)

        s = jnp.clip(s, self.t_eps, 1.0 - self.t_eps)
        t = jnp.clip(s - h, self.t_eps, 1.0)
        mid = jnp.clip(s - h / 2, self.t_eps, 1.0 - self.t_eps)

        vbt = diffrax.VirtualBrownianTree(
            t0=self.t_eps,
            t1=1.0,
            tol=h / 4,
            shape=y0.shape,
            key=vbt_key,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )

        levy_1 = vbt.evaluate(s, mid, use_levy=True)
        y_mid = ema_flow_map(ys, s, mid, levy_1.W, levy_1.H, levy_1.K)  # pyright: ignore
        levy_2 = vbt.evaluate(mid, t, use_levy=True)
        yt_target = ema_flow_map(y_mid, mid, t, levy_2.W, levy_2.H, levy_2.K)  # pyright: ignore
        yt_target = jax.lax.stop_gradient(yt_target)

        levy = vbt.evaluate(s, t, use_levy=True)
        yt_pred = flow_map(ys, s, t, levy.W, levy.H, levy.K, key=dropout_key)  # pyright: ignore

        raw_loss = jnp.sum((yt_pred - yt_target) ** 2) / h
        u = u_mlp(s, t)
        return jnp.exp(-u) * raw_loss + u

    def value(
        self,
        flow_map: AbstractFlowMap,
        ema_flow_map: AbstractFlowMap,
        y0_batch: BatchY,
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        batch_size = y0_batch.shape[0]
        key_h, key_s, key_vbt = jax.random.split(key, 3)

        h_vals = jax.random.uniform(
            key_h, (batch_size,), minval=self.dt, maxval=self.h_max
        )
        s_vals = jax.random.uniform(
            key_s, (batch_size,), minval=self.t_eps + h_vals, maxval=1.0
        )
        vbt_keys = jax.random.split(key_vbt, batch_size)

        losses = eqx.filter_vmap(
            lambda y0, s, h, k: self._single(flow_map, ema_flow_map, y0, s, h, k, u_mlp)
        )(y0_batch, s_vals, h_vals, vbt_keys)

        return jnp.mean(losses)


class UncertaintyJointLoss(eqx.Module):
    score_loss: UncertaintyScoreLoss
    distill_loss: UncertaintyDistillationLoss
    u_mlp: UncertaintyMLP
    eta: float = 0.75

    def value_and_grad(self, flow_map, ema_flow_map, y0_batch, key):
        @eqx.filter_value_and_grad
        def _loss(diff_args):
            fm, u = diff_args
            key_score, key_distill = jax.random.split(key)
            n_score = int(y0_batch.shape[0] * self.eta)
            l_score = self.score_loss.value(
                fm, ema_flow_map, y0_batch[:n_score], key_score, u
            )
            l_distill = self.distill_loss.value(
                fm, ema_flow_map, y0_batch[n_score:], key_distill, u
            )
            return self.eta * l_score + (1 - self.eta) * l_distill

        loss, (fm_grads, u_grads) = _loss((flow_map, self.u_mlp))
        return loss, fm_grads, u_grads
