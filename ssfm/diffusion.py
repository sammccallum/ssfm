from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ssfm.typing import Y


class AbstractDiffusion(eqx.Module):
    reverse_eta: float

    @abstractmethod
    def forward_drift(self, t: Float[Array, ""], y: Y) -> Y:
        pass

    @abstractmethod
    def forward_diffusion(self, t: Float[Array, ""]) -> Float[Array, ""]:
        pass

    @abstractmethod
    def forward_sample(
        self, t: Float[Array, ""], y0: Y, key: PRNGKeyArray
    ) -> tuple[Y, Y]:
        pass

    def reverse_drift(
        self,
        t: Float[Array, ""],
        y: Y,
        score: Y,
    ) -> Y:
        g = self.forward_diffusion(t)
        return self.forward_drift(t, y) - 0.5 * (1 + self.reverse_eta**2) * g**2 * score

    def reverse_diffusion(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return self.reverse_eta * self.forward_diffusion(t)


class VPDiffusion(AbstractDiffusion):
    beta_min: float
    beta_max: float

    def _beta(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _log_mean_coeff(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)

    def forward_drift(self, t: Float[Array, ""], y: Y) -> Y:
        return -0.5 * self._beta(t) * y

    def forward_diffusion(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.sqrt(self._beta(t))

    def forward_sample(
        self, t: Float[Array, ""], y0: Y, key: PRNGKeyArray
    ) -> tuple[Y, Y]:
        alpha = jnp.exp(self._log_mean_coeff(t))
        sigma = jnp.sqrt(1.0 - alpha**2)
        noise = jax.random.normal(key, y0.shape)
        yt = alpha * y0 + sigma * noise
        score = -noise / sigma
        return yt, score
