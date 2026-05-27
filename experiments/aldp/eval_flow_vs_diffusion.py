import json
import math
import os

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from baseline_pmf import sample_iid_reverse_sde
from matplotlib.colors import LogNorm
from scoremd.data.dataset.aldp import ALDPDataset, CoarseGrainingLevel
from scoremd.data.preprocess import CenterMolecule
from scoremd.models import GraphTransformerModelInfo, RangedModel
from scoremd.models.mixture import MixtureOfModels
from scoremd.training.weighting import construct_ranged_constant_weighting_function
from scoremd.utils.evaluation import js_divergence, phi_psi_metrics

from ssfm.diffusion import VPDiffusion
from ssfm.flow_map import EulerMaruyamaFlowMap
from ssfm.graph_transformer import GraphTransformerEMStepModel

FLOW_MAP_CKPT = "experiments/aldp/models/aldp_v1_from_scratch.eqx"

ALDP_VARIANT_CKPTS = {
    "both": "/data/sm2942/projects/ssfm/ScoreMD/models/aldp/both/model",
    "mixture": "/data/sm2942/projects/ssfm/ScoreMD/models/aldp/mixture/model",
    "diffusion": "/data/sm2942/projects/ssfm/ScoreMD/models/aldp/diffusion/model",
    "fp": "/data/sm2942/projects/ssfm/ScoreMD/models/aldp/fp/model",
    "two_for_one": "/data/sm2942/projects/ssfm/ScoreMD/models/aldp/two_for_one/model",
}

ALL_VARIANTS = list(ALDP_VARIANT_CKPTS.keys())


def _aldp_ranged_models(variant: str) -> list[RangedModel]:
    if variant in ("both", "mixture"):
        score = GraphTransformerModelInfo(potential=False)
        potential = GraphTransformerModelInfo(potential=True)
        return [
            RangedModel(score, range=(1.0, 0.6)),
            RangedModel(score, range=(0.6, 0.1)),
            RangedModel(potential, range=(0.1, 0.0)),
        ]
    if variant in ("diffusion", "fp", "two_for_one"):
        potential = GraphTransformerModelInfo(potential=True)
        return [RangedModel(potential, range=(1.0, 0.0))]
    raise ValueError(f"Unknown ALDP variant: {variant}")


def build_aldp_variant(dataset: ALDPDataset, variant: str):
    norm_factor = 1.0 / dataset.std
    ranged = _aldp_ranged_models(variant)
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

    ckpt = ALDP_VARIANT_CKPTS[variant]
    with ocp.CheckpointManager(os.path.abspath(ckpt)) as mgr:
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

    return score_fn


def load_flow_map(
    ckpt_path: str,
    n_atoms: int,
    *,
    hidden_nf: int = 96,
    n_layers: int = 3,
    heads: int = 8,
    dim_head: int = 64,
    ff_mult: int = 4,
    time_dim: int = 64,
    dropout_rate: float = 0.0,
    seed: int = 0,
) -> EulerMaruyamaFlowMap:
    step_model = GraphTransformerEMStepModel(
        n_atoms=n_atoms,
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        heads=heads,
        dim_head=dim_head,
        ff_mult=ff_mult,
        time_dim=time_dim,
        dropout_rate=dropout_rate,
        key=jax.random.PRNGKey(seed),
    )
    flow_map = EulerMaruyamaFlowMap(step_model=step_model)
    return eqx.tree_deserialise_leaves(ckpt_path, flow_map)


def sample_flow_map_chunked(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key,
    n_samples: int,
    data_dim: int,
    n_steps: int,
    chunk: int,
) -> np.ndarray:
    flow_map = eqx.nn.inference_mode(flow_map)
    step_size = (1.0 - t_eps) / n_steps
    time_grid = jnp.linspace(1.0, t_eps, n_steps + 1)

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

        def scan_fn(y, i):
            s = jnp.clip(time_grid[i], t_eps, 1.0)
            t = jnp.clip(time_grid[i + 1], t_eps, 1.0)
            levy = vbt.evaluate(s, t, use_levy=True)
            return flow_map(y, s, t, levy.W, levy.H, levy.K), None

        y, _ = jax.lax.scan(scan_fn, y, jnp.arange(n_steps))
        return y

    sample_chunk = jax.jit(jax.vmap(sample_one))

    chunks = []
    drawn = 0
    k = key
    while drawn < n_samples:
        k, k_chunk = jax.random.split(k)
        size = min(chunk, n_samples - drawn)
        keys = jax.random.split(k_chunk, size)
        chunks.append(np.asarray(sample_chunk(keys)))
        drawn += size
        print(f"  drew {drawn}/{n_samples}")
    return np.concatenate(chunks, axis=0)


def sample_diffusion_chunked(
    score_fn,
    diffusion: VPDiffusion,
    key,
    n_samples: int,
    data_dim: int,
    n_steps: int,
    t_eps: float,
    chunk: int,
) -> np.ndarray:
    chunks = []
    drawn = 0
    k = key
    while drawn < n_samples:
        k, k_chunk = jax.random.split(k)
        size = min(chunk, n_samples - drawn)
        s = sample_iid_reverse_sde(
            score_fn, diffusion, size, data_dim, n_steps, t_eps, k_chunk
        )
        chunks.append(np.asarray(s))
        drawn += size
        print(f"  drew {drawn}/{n_samples}")
    return np.concatenate(chunks, axis=0)


def compute_metrics(
    samples_phys,
    target_phi: np.ndarray,
    target_psi: np.ndarray,
    dataset: ALDPDataset,
    n_bins: int = 64,
) -> tuple[float, float, float]:
    sampled_phi, sampled_psi = dataset.get_2d_features(jnp.asarray(samples_phys))
    sampled_phi = np.asarray(sampled_phi)
    sampled_psi = np.asarray(sampled_psi)
    rms_fe_sq, rms_mjs = phi_psi_metrics(
        target_phi, target_psi, sampled_phi, sampled_psi, n_bins=n_bins
    )
    target_pp = jnp.stack([jnp.asarray(target_phi), jnp.asarray(target_psi)], axis=1)
    sampled_pp = jnp.stack([jnp.asarray(sampled_phi), jnp.asarray(sampled_psi)], axis=1)
    js = js_divergence(target_pp, sampled_pp, bins=n_bins)
    return float(rms_fe_sq), float(rms_mjs), float(js)


_PI_TICKS = [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]
_PI_TICK_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]


def _apply_latex_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titlepad": 8,
        }
    )


def plot_phi_psi_grid(
    samples_by_n_steps: dict[int, np.ndarray],
    dataset: ALDPDataset,
    norm_factor: float,
    target_phi: np.ndarray,
    target_psi: np.ndarray,
    bins: int = 100,
) -> plt.Figure:
    _apply_latex_style()
    n_steps_list = list(samples_by_n_steps.keys())
    n_panels = 1 + len(n_steps_list)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.4 * n_panels, 3.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    def _draw(ax, phi: np.ndarray, psi: np.ndarray, title: str) -> None:
        ax.hist2d(
            phi,
            psi,
            bins=bins,
            range=[[-math.pi, math.pi], [-math.pi, math.pi]],
            density=True,
            norm=LogNorm(),
            rasterized=True,
        )
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_aspect("equal")
        ax.set_xticks(_PI_TICKS)
        ax.set_xticklabels(_PI_TICK_LABELS)
        ax.set_yticks(_PI_TICKS)
        ax.set_yticklabels(_PI_TICK_LABELS)
        ax.set_xlabel(r"$\varphi$")
        ax.set_title(title)

    _draw(axes[0], np.asarray(target_phi), np.asarray(target_psi), "Ground truth")
    axes[0].set_ylabel(r"$\psi$")

    for ax, n_steps in zip(axes[1:], n_steps_list):
        samples_phys = samples_by_n_steps[n_steps] / norm_factor
        phi, psi = dataset.get_2d_features(jnp.asarray(samples_phys))
        _draw(ax, np.asarray(phi), np.asarray(psi), f"{n_steps}-Steps")
    return fig


def plot_phi_psi_grid_with_diffusion(
    ssfm_samples_by_n_steps: dict[int, np.ndarray],
    diffusion_samples_by_n_steps: dict[int, np.ndarray],
    dataset: ALDPDataset,
    norm_factor: float,
    target_phi: np.ndarray,
    target_psi: np.ndarray,
    bins: int = 100,
) -> plt.Figure:
    _apply_latex_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    n_steps_list = list(ssfm_samples_by_n_steps.keys())
    n_model_cols = len(n_steps_list)
    n_cols = 1 + n_model_cols

    fig = plt.figure(
        figsize=(2.6 + 1.7 * n_model_cols, 3.6),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        2, n_cols, width_ratios=[1.4] + [1.0] * n_model_cols, hspace=0.0
    )
    fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0, w_pad=0.02)

    def _draw(
        ax,
        phi: np.ndarray,
        psi: np.ndarray,
        title: str,
        *,
        show_xtick_labels: bool = True,
        show_ytick_labels: bool = True,
        show_xlabel: bool = True,
        ylabel: str | None = None,
    ) -> None:
        ax.hist2d(
            phi,
            psi,
            bins=bins,
            range=[[-math.pi, math.pi], [-math.pi, math.pi]],
            density=True,
            norm=LogNorm(),
            rasterized=True,
        )
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_aspect("equal")
        ax.set_xticks(_PI_TICKS)
        ax.set_yticks(_PI_TICKS)
        ax.set_xticklabels(_PI_TICK_LABELS if show_xtick_labels else [])
        ax.set_yticklabels(_PI_TICK_LABELS if show_ytick_labels else [])
        if show_xlabel:
            ax.set_xlabel(r"$\varphi$")
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

    ax_gt = fig.add_subplot(gs[:, 0])
    _draw(
        ax_gt,
        np.asarray(target_phi),
        np.asarray(target_psi),
        "Ground truth",
        ylabel=r"$\psi$",
    )

    for i, n_steps in enumerate(n_steps_list):
        ax = fig.add_subplot(gs[0, i + 1])
        samples_phys = diffusion_samples_by_n_steps[n_steps] / norm_factor
        phi, psi = dataset.get_2d_features(jnp.asarray(samples_phys))
        _draw(
            ax,
            np.asarray(phi),
            np.asarray(psi),
            f"$n={n_steps}$",
            show_xtick_labels=False,
            show_ytick_labels=False,
            show_xlabel=False,
            ylabel="Diffusion" if i == 0 else None,
        )

    for i, n_steps in enumerate(n_steps_list):
        ax = fig.add_subplot(gs[1, i + 1])
        samples_phys = ssfm_samples_by_n_steps[n_steps] / norm_factor
        phi, psi = dataset.get_2d_features(jnp.asarray(samples_phys))
        _draw(
            ax,
            np.asarray(phi),
            np.asarray(psi),
            "",
            show_xtick_labels=False,
            show_ytick_labels=False,
            show_xlabel=False,
            ylabel="SSFM" if i == 0 else None,
        )

    return fig


def main():
    print(f"JAX devices: {jax.devices()}")

    flow_map_ckpt = os.environ.get("FLOW_MAP_CKPT", FLOW_MAP_CKPT)
    n_samples = int(os.environ.get("N_SAMPLES", "1024"))
    chunk = int(os.environ.get("CHUNK", "4000"))
    seed = int(os.environ.get("SEED", "42"))
    t_eps = 1e-5
    n_steps_env = os.environ.get("N_STEPS_LIST", "")
    if n_steps_env:
        n_steps_list = [int(x) for x in n_steps_env.split(",")]
    else:
        n_steps_list = [2, 4, 10, 20, 100, 1000]
    n_bins = 64

    variants_env = os.environ.get("VARIANTS")
    if variants_env is None:
        variants = ALL_VARIANTS
    else:
        variants = [v.strip() for v in variants_env.split(",") if v.strip()]
    for v in variants:
        if v not in ALDP_VARIANT_CKPTS:
            raise ValueError(
                f"Unknown variant {v!r}; valid: {list(ALDP_VARIANT_CKPTS)}"
            )
    run_ssfm = os.environ.get("RUN_SSFM", "1") != "0"

    plot_path_png = os.environ.get(
        "PLOT_PATH_PNG", "experiments/aldp/eval_flow_phi_psi.png"
    )
    plot_path_pdf = os.environ.get(
        "PLOT_PATH_PDF", "experiments/aldp/eval_flow_phi_psi.pdf"
    )
    results_path = os.environ.get("RESULTS_PATH", "experiments/aldp/eval_results.json")

    print("Loading ALDP dataset...")
    dataset = ALDPDataset(
        coarse_graining_level=CoarseGrainingLevel.FULL,
        limit_samples=50_000,
        validation=False,
        seed=0,
    )
    data_dim = dataset.train.data.shape[1]
    n_atoms = data_dim // 3
    norm_factor = 1.0 / dataset.std
    print(
        f"  data_dim={data_dim}, n_atoms={n_atoms}, "
        f"n_train={dataset.train.data.shape[0]}, norm_factor={norm_factor:.4f}"
    )

    target_phi, target_psi = dataset.get_2d_features(dataset.train.data)
    target_phi = np.asarray(target_phi)
    target_psi = np.asarray(target_psi)

    diffusion = VPDiffusion(beta_min=0.1, beta_max=20.0, reverse_eta=1.0)

    key = jax.random.PRNGKey(seed)
    n_routes = (1 if run_ssfm else 0) + len(variants)
    route_keys = jax.random.split(key, max(n_routes, 1))
    key_idx = 0

    results: dict[str, dict[int, tuple[float, float, float]]] = {}
    flow_samples_by_n: dict[int, np.ndarray] = {}
    mixture_samples_by_n: dict[int, np.ndarray] = {}

    if run_ssfm:
        print(f"\n### SSFM ### loading from {flow_map_ckpt}...")
        flow_map = load_flow_map(flow_map_ckpt, n_atoms)
        flow_results: dict[int, tuple[float, float, float]] = {}
        flow_step_keys = jax.random.split(route_keys[key_idx], len(n_steps_list))
        key_idx += 1
        for n_steps, k in zip(n_steps_list, flow_step_keys):
            print(f"\n=== SSFM | n_steps = {n_steps} ===")
            samples = sample_flow_map_chunked(
                flow_map, t_eps, k, n_samples, data_dim, n_steps, chunk
            )
            flow_samples_by_n[n_steps] = samples
            pmf, mjs, js = compute_metrics(
                samples / norm_factor, target_phi, target_psi, dataset, n_bins
            )
            flow_results[n_steps] = (pmf, mjs, js)
            print(
                f"SSFM n_steps={n_steps:>4d} | PMF={pmf:.4f} | "
                f"MJS={mjs:.4f} | JS={js:.4f}"
            )
        results["ssfm"] = flow_results

    for variant in variants:
        print(f"\n### {variant} ### building score_fn from checkpoint...")
        score_fn = build_aldp_variant(dataset, variant)
        var_step_keys = jax.random.split(route_keys[key_idx], len(n_steps_list))
        key_idx += 1
        var_results: dict[int, tuple[float, float, float]] = {}
        for n_steps, k in zip(n_steps_list, var_step_keys):
            print(f"\n=== {variant} | n_steps = {n_steps} ===")
            samples = sample_diffusion_chunked(
                score_fn, diffusion, k, n_samples, data_dim, n_steps, t_eps, chunk
            )
            if variant == "mixture":
                mixture_samples_by_n[n_steps] = samples
            pmf, mjs, js = compute_metrics(
                samples / norm_factor, target_phi, target_psi, dataset, n_bins
            )
            var_results[n_steps] = (pmf, mjs, js)
            print(
                f"{variant:>11s} n_steps={n_steps:>4d} | PMF={pmf:.4f} | "
                f"MJS={mjs:.4f} | JS={js:.4f}"
            )
        results[variant] = var_results

    if run_ssfm:
        if all(n in mixture_samples_by_n for n in flow_samples_by_n):
            fig = plot_phi_psi_grid_with_diffusion(
                flow_samples_by_n,
                mixture_samples_by_n,
                dataset,
                norm_factor,
                target_phi,
                target_psi,
            )
        else:
            fig = plot_phi_psi_grid(
                flow_samples_by_n, dataset, norm_factor, target_phi, target_psi
            )
        os.makedirs(os.path.dirname(plot_path_png) or ".", exist_ok=True)
        fig.savefig(plot_path_png, dpi=200, bbox_inches="tight")
        fig.savefig(plot_path_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved flow-map phi/psi grid -> {plot_path_png}, {plot_path_pdf}")

    print("\nSummary (vs ALDP training distribution, n_bins=64):")
    print(f"  {'route':<12s} | {'n_steps':>8s} | {'PMF':>10s} {'MJS':>10s} {'JS':>10s}")
    print("  " + "-" * 60)
    for route, route_results in results.items():
        for n_steps in n_steps_list:
            if n_steps not in route_results:
                continue
            pmf, mjs, js = route_results[n_steps]
            print(
                f"  {route:<12s} | {n_steps:>8d} | "
                f"{pmf:>10.4f} {mjs:>10.4f} {js:>10.4f}"
            )

    serialisable = {
        route: {str(n): list(metrics) for n, metrics in r.items()}
        for route, r in results.items()
    }
    payload = {
        "config": {
            "n_samples": n_samples,
            "n_steps_list": n_steps_list,
            "n_bins": n_bins,
            "t_eps": t_eps,
            "seed": seed,
            "flow_map_ckpt": flow_map_ckpt,
            "variants": variants,
            "run_ssfm": run_ssfm,
        },
        "metric_order": ["PMF", "MJS", "JS"],
        "results": serialisable,
    }
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing = json.load(f)
            if (
                existing.get("config", {}).get("n_samples") == n_samples
                and existing.get("config", {}).get("n_steps_list") == n_steps_list
            ):
                merged = existing.get("results", {})
                merged.update(serialisable)
                payload["results"] = merged
                payload["config"]["variants"] = sorted(
                    set(existing.get("config", {}).get("variants", [])) | set(variants)
                )
                payload["config"]["run_ssfm"] = (
                    existing.get("config", {}).get("run_ssfm", False) or run_ssfm
                )
        except (json.JSONDecodeError, OSError):
            pass
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results -> {results_path}")


if __name__ == "__main__":
    main()
