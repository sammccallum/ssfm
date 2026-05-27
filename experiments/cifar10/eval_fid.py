import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from main import (
    cifar10,
    sample_flow_map_batched,
)

from ssfm.edm2_unet import EDM2_UNet, EDM2EMStepModel
from ssfm.flow_map import EulerMaruyamaFlowMap
from ssfm.metrics import FIDStats, compute_fid
from ssfm.metrics.fid import compute_statistics
from ssfm.metrics.inception import InceptionFeatureExtractor
from ssfm.metrics.preprocessing import preprocess_images


def compute_and_cache_real_stats(
    dataset: jnp.ndarray,
    cache_path: str,
    batch_size: int = 64,
) -> FIDStats:
    if os.path.exists(cache_path):
        print(f"Loading cached real stats from {cache_path}")
        return FIDStats.load(cache_path)

    print("Computing Inception stats for real dataset...")
    images = np.asarray(dataset)
    extractor = InceptionFeatureExtractor()
    all_features = []
    for batch in preprocess_images(images, batch_size=batch_size):
        all_features.append(extractor.features(batch))
    features = np.concatenate(all_features, axis=0)
    stats = compute_statistics(features)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    stats.save(cache_path)
    print(f"Saved real stats to {cache_path}")
    return stats


def build_model(key, *, in_channels=3, resolution=32):
    key_drift, key_diff = jax.random.split(key)

    base_channels = 128
    channel_mult = (2, 2, 2)
    num_groups = 8
    dropout_rate = 0.13
    attn_resolutions = [16]
    head_dim = 64
    num_res_blocks = 4

    diff_base_channels = 64
    diff_channel_mult = (2, 2, 2)
    diff_attn_resolutions = [16]

    drift_net = EDM2_UNet(
        in_channels=4 * in_channels,
        out_channels=in_channels,
        resolution=resolution,
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_groups=num_groups,
        dropout_rate=dropout_rate,
        attn_resolutions=attn_resolutions,
        head_dim=head_dim,
        num_res_blocks=num_res_blocks,
        key=key_drift,
    )

    diffusion_net = EDM2_UNet(
        in_channels=3 * in_channels,
        out_channels=in_channels,
        resolution=resolution,
        base_channels=diff_base_channels,
        channel_mult=diff_channel_mult,
        num_groups=num_groups,
        dropout_rate=0,
        attn_resolutions=diff_attn_resolutions,
        head_dim=head_dim,
        num_res_blocks=num_res_blocks,
        key=key_diff,
    )

    step_model = EDM2EMStepModel(drift_net=drift_net, diffusion_net=diffusion_net)
    return EulerMaruyamaFlowMap(step_model=step_model)


def main(
    model_path="experiments/cifar10/models/cifar10_edm2_em.eqx",
    checkpoint_path="experiments/cifar10/checkpoints/400k",
    n_samples=50_000,
    sample_batch_size=256,
    step_sizes=(1 / 16, 1 / 8, 1 / 4, 1 / 2),
    t_eps=1e-5,
    seed=42,
):
    in_channels = 3
    resolution = 32
    data_shape = (in_channels, resolution, resolution)

    dataset, data_mean, data_std = cifar10()
    dataset = dataset.reshape(-1, *data_shape)
    data_mean = data_mean.reshape(data_shape)
    data_std = data_std.reshape(data_shape)

    real_stats_path = "experiments/cifar10/data/cifar10_inception_stats.npz"
    dataset_for_fid = dataset * data_std[None] + data_mean[None]
    dataset_for_fid = dataset_for_fid * 2.0 - 1.0
    real_stats = compute_and_cache_real_stats(
        dataset_for_fid, cache_path=real_stats_path
    )

    key = jax.random.PRNGKey(0)
    flow_map = build_model(key, in_channels=in_channels, resolution=resolution)
    if checkpoint_path is not None:
        ema_path = os.path.join(checkpoint_path, "ema_flow_map.eqx")
        flow_map = eqx.tree_deserialise_leaves(ema_path, flow_map)
        print(f"Loaded EMA model from checkpoint {checkpoint_path}")
    else:
        flow_map = eqx.tree_deserialise_leaves(model_path, flow_map)
        print(f"Loaded model from {model_path}")

    key = jax.random.PRNGKey(seed)
    for ss in step_sizes:
        n_steps = max(1, math.ceil((1.0 - t_eps) / ss))
        key, key_fid = jax.random.split(key)
        fid_samples = sample_flow_map_batched(
            flow_map,
            t_eps,
            key_fid,
            n_samples,
            data_shape,
            ss,
            n_steps,
            batch_size=sample_batch_size,
        )
        fid_samples = fid_samples * data_std[None] + data_mean[None]
        fid_samples = jnp.clip(fid_samples, 0.0, 1.0)
        fid_samples = fid_samples * 2.0 - 1.0
        fid = compute_fid(np.array(fid_samples), real_stats, batch_size=64)
        print(f"FID (h={ss}, steps={n_steps}): {fid:.2f}")


if __name__ == "__main__":
    main()
