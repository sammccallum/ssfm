import os

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from scoremd.data.dataset.aldp import ALDPDataset, CoarseGrainingLevel
from scoremd.data.preprocess import CenterMolecule
from scoremd.evaluate.molecules import simulate_molecule
from scoremd.models import GraphTransformerModelInfo, RangedModel
from scoremd.models.mixture import MixtureOfModels
from scoremd.training.weighting import construct_ranged_constant_weighting_function
from scoremd.utils.evaluation import js_divergence, phi_psi_metrics

from ssfm.diffusion import VPDiffusion

ALDP_BOTH_CKPT = "/data/sm2942/projects/ssfm/ScoreMD/models/aldp/both/model"


def build_aldp_both_model(dataset):
    norm_factor = 1.0 / dataset.std
    score_info = GraphTransformerModelInfo(potential=False)
    potential_info = GraphTransformerModelInfo(potential=True)
    ranged = [
        RangedModel(score_info, range=(1.0, 0.6)),
        RangedModel(score_info, range=(0.6, 0.1)),
        RangedModel(potential_info, range=(0.1, 0.0)),
    ]
    ranged = sorted(ranged, key=lambda m: m.range[0], reverse=True)
    weight = construct_ranged_constant_weighting_function(ranged, normalize=True)
    model = MixtureOfModels(
        [m.build(dataset, norm_factor) for m in ranged],
        weight,
        CenterMolecule(dataset),
    )

    data_dim = dataset.train.data.shape[1]
    abstract = model.init(
        jax.random.PRNGKey(0),
        jnp.ones([1, data_dim]),
        None,
        jnp.ones([1, 1]),
        training=False,
    )

    with ocp.CheckpointManager(os.path.abspath(ALDP_BOTH_CKPT)) as mgr:
        restored = mgr.restore(
            mgr.latest_step(),
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(abstract),
                ema_params=ocp.args.StandardRestore(abstract),
            ),
        )
    ema_params = restored.ema_params

    def score_fn(y, t):
        return model.apply(ema_params, y, None, t, training=False)

    return model, ema_params, score_fn, norm_factor


def sample_iid_reverse_sde(
    score_fn, diffusion, n_samples, data_dim, n_steps, t_eps, key
):
    time_grid = jnp.linspace(1.0, t_eps, n_steps + 1)
    step_size = (1.0 - t_eps) / n_steps

    def sample_one(k):
        k_init, k_vbt = jax.random.split(k)
        y = jax.random.normal(k_init, (data_dim,))
        vbt = diffrax.VirtualBrownianTree(
            t0=t_eps,
            t1=1.0,
            tol=step_size / 4,
            shape=(data_dim,),
            key=k_vbt,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )

        def step(y, i):
            s = time_grid[i]
            t = time_grid[i + 1]
            score = score_fn(y, s)
            drift = diffusion.reverse_drift(s, y, score)
            g = diffusion.reverse_diffusion(s)
            levy = vbt.evaluate(s, t, use_levy=True)
            return y + drift * (t - s) + g * levy.W, None

        y, _ = jax.lax.scan(step, y, jnp.arange(n_steps))
        return y

    keys = jax.random.split(key, n_samples)
    return jax.vmap(sample_one)(keys)


def compute_pmf(samples_phys, target_phi, target_psi, dataset, n_bins=64):
    sampled_phi, sampled_psi = dataset.get_2d_features(samples_phys)
    sampled_phi = np.asarray(sampled_phi)
    sampled_psi = np.asarray(sampled_psi)
    rms_fe_sq, rms_mjs = phi_psi_metrics(
        target_phi,
        target_psi,
        sampled_phi,
        sampled_psi,
        n_bins=n_bins,
    )
    target_phi_psi = jnp.stack(
        [jnp.asarray(target_phi), jnp.asarray(target_psi)], axis=1
    )
    sampled_phi_psi = jnp.stack(
        [jnp.asarray(sampled_phi), jnp.asarray(sampled_psi)], axis=1
    )
    js = js_divergence(target_phi_psi, sampled_phi_psi, bins=n_bins)
    return float(rms_fe_sq), float(rms_mjs), float(js)


def run_langevin(
    model,
    ema_params,
    norm_factor,
    dataset,
    n_parallel,
    n_samples,
    n_intermediate_steps,
    langevin_dt,
    eval_t,
    ensure_low_prob,
    seed,
):
    def force(x, features, **kwargs):
        return (
            dataset.kbT
            * model.apply(
                ema_params,
                x * norm_factor,
                features,
                eval_t,
                training=False,
                method=model.__class__.force,
            )
            * norm_factor
        )

    data = dataset.train.data
    if data.shape[0] < n_parallel:
        data = jnp.repeat(data, n_parallel, axis=0)
    initial_positions = data[:n_parallel]

    if ensure_low_prob:
        gt_phi, gt_psi = dataset.get_2d_features(data)
        is_low = (gt_phi > 0.0) & (gt_phi < 2.0) & (gt_psi > -2) & (gt_psi < 2)
        if bool(jnp.any(is_low)):
            first_low = data[is_low][0]
            initial_positions = jnp.concatenate(
                [first_low[None], initial_positions[:-1]], axis=0
            )
            print("  seeded chain 0 in the high-energy basin")
        else:
            print("  no low-probability state found in dataset")

    trajectories, _ = simulate_molecule(
        dataset,
        force,
        initial_positions,
        None,
        n_samples,
        n_intermediate_steps,
        langevin_dt,
        seed,
    )
    return trajectories


def main():
    print(f"JAX devices: {jax.devices()}")

    n_iid_samples = int(os.environ.get("N_IID", "50000"))
    n_steps_list = [100, 20, 10, 4, 2]
    t_eps = 1e-5
    chunk = 4096

    n_langevin_parallel = int(os.environ.get("N_CHAINS", "100"))
    n_per_chain = int(os.environ.get("N_PER_CHAIN", "1024"))
    n_intermediate_steps = int(os.environ.get("N_INT", "50"))
    langevin_dt = float(os.environ.get("LANGEVIN_DT", "0.002"))
    eval_t = float(os.environ.get("EVAL_T", "1e-5"))

    print("Loading ALDP dataset...")
    dataset = ALDPDataset(
        coarse_graining_level=CoarseGrainingLevel.FULL,
        limit_samples=50_000,
        validation=False,
        seed=0,
    )
    data_dim = dataset.train.data.shape[1]
    print(f"  data_dim={data_dim}, n_train={dataset.train.data.shape[0]}")

    print("Building score_fn from pretrained checkpoint...")
    model, ema_params, score_fn, norm_factor = build_aldp_both_model(dataset)
    print(f"  norm_factor = {norm_factor:.4f}")

    target_phi, target_psi = dataset.get_2d_features(dataset.train.data)
    target_phi = np.asarray(target_phi)
    target_psi = np.asarray(target_psi)

    diffusion = VPDiffusion(beta_min=0.1, beta_max=20.0, reverse_eta=1.0)

    seed = int(os.environ.get("SEED", "42"))
    key = jax.random.PRNGKey(seed)
    key_iid, key_lang = jax.random.split(key)

    iid_results = {}
    iid_step_keys = jax.random.split(key_iid, len(n_steps_list))
    for n_steps, k_steps in zip(n_steps_list, iid_step_keys):
        print(
            f"Sampling {n_iid_samples} iid in chunks of {chunk} "
            f"({n_steps}-step reverse SDE, seed={seed})..."
        )
        chunks = []
        drawn = 0
        k = k_steps
        while drawn < n_iid_samples:
            k, k_chunk = jax.random.split(k)
            size = min(chunk, n_iid_samples - drawn)
            s = sample_iid_reverse_sde(
                score_fn, diffusion, size, data_dim, n_steps, t_eps, k_chunk
            )
            chunks.append(np.asarray(s))
            drawn += size
            print(f"  drew {drawn}/{n_iid_samples}")
        iid_samples = np.concatenate(chunks, axis=0)
        iid_phys = iid_samples / norm_factor

        iid_pmf, iid_mjs, iid_js = compute_pmf(
            jnp.asarray(iid_phys), target_phi, target_psi, dataset
        )
        iid_results[n_steps] = (iid_pmf, iid_mjs, iid_js)
        print(
            f"IID n_steps={n_steps:>4d} | PMF={iid_pmf:.4f}, "
            f"MJS={iid_mjs:.4f}, JS={iid_js:.4f}"
        )

    print("IID summary:")
    for n_steps, (pmf, mjs, js) in iid_results.items():
        print(f"  n_steps={n_steps:>4d}: PMF={pmf:.4f}, MJS={mjs:.4f}, JS={js:.4f}")

    print(
        f"Running Langevin: {n_langevin_parallel} chains x {n_per_chain} stored samples "
        f"({n_intermediate_steps} intermediate steps each, "
        f"{n_per_chain * n_intermediate_steps * 0.001:.1f} ps per chain)"
    )
    lang_seed = int(jax.random.randint(key_lang, (), 0, 2**30))
    trajectories = run_langevin(
        model,
        ema_params,
        norm_factor,
        dataset,
        n_langevin_parallel,
        n_per_chain,
        n_intermediate_steps,
        langevin_dt,
        eval_t,
        ensure_low_prob=True,
        seed=lang_seed,
    )
    trajectories_np = np.asarray(trajectories)
    n_nan = int(np.isnan(trajectories_np).any(axis=(1, 2)).sum())
    print(f"  trajectories.shape = {trajectories.shape}, NaN chains = {n_nan}")

    out_path = os.environ.get("LANG_TRAJ_PATH", "")
    if out_path:
        np.save(out_path, trajectories_np)
        print(f"  saved trajectories -> {out_path}")

    lang_flat = jnp.asarray(trajectories).reshape(-1, data_dim)
    lang_pmf, lang_mjs, lang_js = compute_pmf(
        lang_flat, target_phi, target_psi, dataset
    )
    print(f"LANG | PMF={lang_pmf:.4f}, MJS={lang_mjs:.4f}, JS={lang_js:.4f}")

    for i, t in enumerate(np.asarray(trajectories)):
        phi, psi = dataset.get_2d_features(jnp.asarray(t))
        phi = np.asarray(phi)
        psi = np.asarray(psi)
        in_low = ((phi > 0.0) & (phi < 2.0) & (psi > -2) & (psi < 2)).mean()
        print(
            f"  chain {i}: phi range=({phi.min():+.2f}, {phi.max():+.2f}), "
            f"psi range=({psi.min():+.2f}, {psi.max():+.2f}), frac in high-E basin={in_low:.3f}"
        )


if __name__ == "__main__":
    main()
