from abc import abstractmethod

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from ssfm.typing import Y


class AbstractEMStepModel(eqx.Module):
    @abstractmethod
    def drift(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        pass

    @abstractmethod
    def diffusion(
        self,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        pass


class AbstractFlowMap(eqx.Module):
    @abstractmethod
    def __call__(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        pass


class EulerMaruyamaFlowMap(AbstractFlowMap):
    step_model: AbstractEMStepModel

    def __call__(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        if key is not None:
            key_drift, key_diff = jax.random.split(key)
        else:
            key_drift = key_diff = None
        return (
            ys
            + (t - s) * self.step_model.drift(ys, s, t, W, H, K, key=key_drift)
            + W * self.step_model.diffusion(s, t, W, H, K, key=key_diff)
        )
