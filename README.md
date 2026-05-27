# Strong Stochastic Flow Maps

Reference implementation for the paper *Strong Stochastic Flow Maps* (preprint
link TBD). `ssfm/` is the library; `experiments/` holds the scripts that
reproduce the results in the paper.

## Repository layout

```
ssfm/             # library: flow maps, diffusions, losses, DiT / EDM2-UNet backbones
experiments/
  sde/            # non-linear SDE: strong-order convergence study and ablation
  cifar10/        # CIFAR-10 image generation (EDM2-UNet flow map)
  celeba/         # CelebA 64x64 image generation
  aldp/           # alanine dipeptide MD -- see experiments/aldp/README.md
  chignolin/      # chignolin MD (dataset not yet public)
```

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). From the repo
root:

```bash
uv sync                  # creates .venv (Python 3.13) and installs ssfm (editable) + deps
uv sync --extra metrics  # also installs torch/torchvision, needed only for FID evaluation
```

This pulls `jax[cuda12]`, so a CUDA 12 GPU is expected for the image and MD
experiments.

The **aldp** experiment needs a separate Python 3.11 environment (`.venv-md`)
because of its MD dependencies — see [`experiments/aldp/README.md`](experiments/aldp/README.md).

## Running experiments

Run every script **from the repo root** with the project `venv`:

```bash
.venv/bin/python experiments/<name>/<script>.py
```

The working directory must be the repo root — scripts resolve data and output
paths relative to it (e.g. `experiments/cifar10/data/`).

Training scripts log to Weights & Biases (project `Stochastic Flow Map`). Run
`wandb login` first, or set `WANDB_MODE=offline` to skip it.

### SDE

A non-linear SDE, fit with strong stochastic flow maps using polynomial Brownian coefficients of order `N`.

```bash
.venv/bin/python experiments/sde/sde.py --n-coeffs 1   # train + log MSE vs step size to experiments/sde/logs/ (N in 1..4)
.venv/bin/python experiments/sde/main_figure.py        # main-text figure -> experiments/sde/figures/
.venv/bin/python experiments/sde/coarse_errors.py      # error-vs-step-size figure from the logs/
```

### CIFAR-10

```bash
.venv/bin/python experiments/cifar10/main.py            # train (auto-downloads CIFAR-10, shards over all GPUs)
.venv/bin/python experiments/cifar10/eval_fid.py        # FID vs the dataset (needs the metrics extra)
.venv/bin/python experiments/cifar10/bm_consistency.py  # shared-Brownian-motion consistency figure
```

### CelebA (64x64)

```bash
.venv/bin/python experiments/celeba/main.py             # train (downloads CelebA via gdown, caches an .npz)
.venv/bin/python experiments/celeba/eval_fid.py         # FID (needs the metrics extra)
```

### ALDP (alanine dipeptide)

Uses the separate `.venv-md` environment and the ScoreMD library. Full setup and
run instructions are in [`experiments/aldp/README.md`](experiments/aldp/README.md).

### Chignolin

The chignolin experiments are included for reference but are not yet runnable; the MD dataset is not open-source but will be released shortly. The training scripts
(`train_strong_sfm.py`, the `train_xpred_diffusion.py` baseline) are Hydra-based
and expect a `configs/` directory and a coordinate dataset, both of which will
be released alongside the data. The analysis scripts (`sampling.py`,
`tica_results.py`, `torus_results.py`, `xyz_results.py`) operate on trained
checkpoints. These run in the main `.venv`.
