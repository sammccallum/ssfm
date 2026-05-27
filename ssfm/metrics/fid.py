import dataclasses

import numpy as np
from scipy import linalg

from .inception import InceptionFeatureExtractor
from .preprocessing import preprocess_images


@dataclasses.dataclass
class FIDStats:
    mu: np.ndarray
    sigma: np.ndarray
    n_samples: int

    def save(self, path: str) -> None:
        np.savez(path, mu=self.mu, sigma=self.sigma, n_samples=self.n_samples)

    @classmethod
    def load(cls, path: str) -> "FIDStats":
        data = np.load(path)
        return cls(
            mu=data["mu"],
            sigma=data["sigma"],
            n_samples=int(data["n_samples"]),
        )


def compute_statistics(features: np.ndarray) -> FIDStats:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return FIDStats(mu=mu, sigma=sigma, n_samples=features.shape[0])


def frechet_distance(stats1: FIDStats, stats2: FIDStats) -> float:
    mu1, sigma1 = stats1.mu, stats1.sigma
    mu2, sigma2 = stats2.mu, stats2.sigma

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):  # pyright: ignore
            raise ValueError(
                f"Imaginary component too large: {np.max(np.abs(covmean.imag))}"  # pyright: ignore
            )
        covmean = covmean.real  # pyright: ignore

    fid = float(
        diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )
    return fid


def compute_fid(
    generated_samples: np.ndarray,
    real_stats: FIDStats | str,
    batch_size: int = 64,
    device: str | None = None,
) -> float:
    generated_samples = np.asarray(generated_samples)

    if isinstance(real_stats, str):
        real_stats = FIDStats.load(real_stats)

    extractor = InceptionFeatureExtractor(device=device)

    all_features = []
    for batch in preprocess_images(generated_samples, batch_size=batch_size):
        feats = extractor.features(batch)
        all_features.append(feats)
    gen_features = np.concatenate(all_features, axis=0)

    gen_stats = compute_statistics(gen_features)
    return frechet_distance(gen_stats, real_stats)
