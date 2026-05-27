# ALDP (Alanine Dipeptide) Experiments

These experiments use the [ScoreMD](https://github.com/noegroup/ScoreMD)
library for the ALDP dataset, pretrained score-network baselines, and
OpenMM-based Langevin reference simulations.

ScoreMD pins `mdtraj==1.9.9`, which only builds on Python 3.11. The main
`ssfm` venv uses Python 3.13, so we keep MD in a parallel environment at
`.venv-md/`.

## Setup

From the project root:

```bash
# Clone ScoreMD into the project (gitignored).
git clone https://github.com/noegroup/ScoreMD

# Create a Python 3.11 venv for MD.
uv venv --python 3.11 .venv-md

# Install ssfm (with CIFAR metrics extra).
VIRTUAL_ENV=$PWD/.venv-md uv pip install -e ".[metrics]"

# Install ScoreMD (editable) and its runtime deps. Pins match what the
# original ScoreMD pixi environment resolves to.
VIRTUAL_ENV=$PWD/.venv-md uv pip install -e ./ScoreMD \
    "bgmol @ git+https://github.com/noegroup/bgmol.git" \
    "bgflow @ git+https://github.com/noegroup/bgflow.git" \
    "jax[cuda12]==0.6.2" \
    "flax==0.10.4" \
    "orbax-checkpoint==0.11.6" \
    "mdtraj==1.9.9" \
    "openmm" "deeptime" "einops" "nglview" "pandas" \
    "python-dotenv" "tqdm" "tables" "numpy<2"
```

Activate with `source .venv-md/bin/activate` or invoke directly via
`.venv-md/bin/python <script>`.

## Pretrained ScoreMD checkpoints

`baseline_pmf.py` and `eval_flow_vs_diffusion.py` load pretrained ScoreMD
models from `ScoreMD/models/aldp/{both,mixture,diffusion,fp,two_for_one}/`.
Download the release archive and extract it:

```bash
cd ScoreMD
curl -L -O https://github.com/noegroup/ScoreMD/releases/download/1.0.0/models.zip
unzip models.zip
cd ..
```

## ALDP dataset

The dataset auto-downloads (~470 MB) the first time `ALDPDataset(...)` is
instantiated and lands in `storage/AImplicitUnconstrained/`. The `storage/`
directory is gitignored.

## Running the scripts

```bash
# Train a flow map from scratch on ALDP. Logs to wandb.
.venv-md/bin/python experiments/aldp/from_scratch.py

# Run the pretrained-diffusion + Langevin baseline on ALDP.
.venv-md/bin/python experiments/aldp/baseline_pmf.py

# Compare SSFM samples against the pretrained diffusion baselines for
# n_steps in {2, 4, 10, 20, 100, 1000}. Reads the flow-map checkpoint at
# experiments/aldp/models/aldp_v1_from_scratch.eqx and the ScoreMD variants
# under ScoreMD/models/aldp/. Override via env vars (FLOW_MAP_CKPT,
# N_SAMPLES, VARIANTS, ...).
.venv-md/bin/python experiments/aldp/eval_flow_vs_diffusion.py
```
