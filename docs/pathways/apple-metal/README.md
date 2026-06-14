# Pathway: Apple Silicon / Metal

This pathway runs on macOS with Apple Silicon. It now supports a **real
RFdiffusion diffusion stage on Metal (`mps`)** in addition to `--dry-run`
validation.

## What runs where on Apple Silicon

- **RFdiffusion (stage 1)**: real runs on `mps` (Metal GPU) or `cpu` via a
  drop-in community fork — see below.
- **ProteinMPNN (stage 2)**: accepts `--backend mps/cpu` (disables CUDA,
  enables the MPS CPU-fallback).
- **Protenix (stage 3)**: Docker image is **CUDA-only**; `--runtime local
  --backend mps` only works if your local Protenix build supports MPS.
- **FoldX / ranking (stages 4–5)**: CPU-friendly.

So on a Mac you can run stages 1, 2, 4, 5 for real and either dry-run or
cloud/CUDA stage 3.

## Real diffusion on Metal: the MPS fork

Upstream `RosettaCommons/RFdiffusion` is CUDA-only (its SE3-Transformer
dependency is built against CUDA/DGL). The community
[`YaoYinYing/RFdiffusion@mps-test`](https://github.com/YaoYinYing/RFdiffusion/tree/mps-test)
fork mocks out the CUDA bits and enables the Metal (`mps`) device. Crucially
it keeps the **identical `scripts/run_inference.py` Hydra entry point and
flags** (`inference.*`, `contigmap.*`, `diffuser.partial_T`, …), so it is a
drop-in for our normal flow — `scripts/run_rfdiffusion.py` simply points at
this checkout and lets the fork auto-select the device.

### One-time install (conda)

```bash
# Clone the MPS fork where the pipeline expects it (or anywhere + pass
# --rfd-metal-script / --metal_rfdiffusion_script).
git clone -b mps-test https://github.com/YaoYinYing/RFdiffusion ~/Code/RFdiffusion-mps
cd ~/Code/RFdiffusion-mps

# Apple-Silicon env without CUDA.
conda env create -f env/SE3nv_macos.yml
conda activate RFdiffusion

# CPU/MPS PyTorch + DGL (DGL pins to torch 2.3.0).
conda install 'pytorch==2.3.0' torchvision torchaudio cpuonly -c pytorch
# NOTE on Apple Silicon: data.dgl.ai has NO osx-arm64 wheel, so the upstream
# `pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/repo.html` fails with
# "No matching distribution". Use conda-forge, which ships an osx-arm64 build:
conda install -c conda-forge dgl=2.2.1
# (keep torch at 2.3.0 — dgl's graphbolt dylib is named per torch version)

# Mocked NVTX C headers, then the real NVTX python binding.
pip install git+https://github.com/YaoYinYing/nvtx-mock --force-reinstall
pip install nvtx

# SE3-Transformer with CUDA mocked out and MPS enabled, + dllogger.
pip install git+https://github.com/YaoYinYing/SE3Transformer@rfdiffusion-mps-test
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# The MPS RFdiffusion itself.
pip install -e .
pip install pydantic torchdata==0.9.0

# Model weights (same as upstream).
mkdir -p models && cd models
bash ../scripts/download_models.sh .
```

Total setup is ~30 min. Then download the wild-type templates as in the main
README (`templates/6T7B.pdb`, `templates/6VTX.pdb`).

> Requirements: native arm64 Python, macOS 12.3+. Verify Metal with
> `python -c "import torch; print(torch.backends.mps.is_available())"`.

## Two environments: the pipeline venv vs. the RFdiffusion conda env

This is the #1 gotcha. The fork installs into its **own conda env**
(`RFdiffusion`), while the pipeline runs from `retro_pipeline/.venv`. The
stage-1 wrapper shells out to `run_inference.py`, and by default it uses
*whatever Python is running the pipeline* — which is the venv, and the venv
does **not** have `omegaconf`/`torch`/`se3-transformer`. That produces:

```
ModuleNotFoundError: No module named 'omegaconf'
```

Fix it by pointing the wrapper at the conda env's interpreter with
`--rfdiffusion_python` (per-stage) or `--rfd-python` (orchestrator), e.g.
`~/miniconda3/envs/RFdiffusion/bin/python`. (Alternatively, install the
pipeline into the conda env and run everything from there — but the two-env
split is cleaner.)

## Real diffusion run on Metal (orchestrator)

The diffusion stage runs against the fork; downstream stages run on
`mps/cpu` or dry-run as you prefer.

```bash
cd retro_pipeline
source .venv/bin/activate

# Stage 1 (RFdiffusion) for real on Metal; if your local Protenix lacks MPS,
# resume stage 3 later on a CUDA box / cloud.
./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --backend mps \
  --rfd-python ~/miniconda3/envs/RFdiffusion/bin/python \
  --protenix-runtime local \
  --top-n 100
```

The orchestrator defaults the fork location to
`~/Code/RFdiffusion-mps/scripts/run_inference.py`. If you cloned it elsewhere,
point the orchestrator at it:

```bash
./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --backend mps \
  --rfd-metal-script "$HOME/somewhere-else/RFdiffusion-mps/scripts/run_inference.py"
```

## Real diffusion run on Metal (per-stage)

```bash
# Start small — Metal is much slower than an A100, so validate with a few
# designs before scaling num_backbones up.
python -m scripts.run_rfdiffusion \
  --config configs/sox2.yaml \
  --out_dir workspace/01_backbones \
  --backend mps \
  --num_designs 4 \
  --rfdiffusion_python ~/miniconda3/envs/RFdiffusion/bin/python
```

Useful knobs for Metal:

- `--backend mps` selects the Metal GPU; `--backend cpu` forces the fork onto
  CPU (slower but maximally compatible).
- `--rfdiffusion_python PATH` (orchestrator: `--rfd-python`) — the interpreter
  that has RFdiffusion + its deps; point it at the fork's conda env, e.g.
  `~/miniconda3/envs/RFdiffusion/bin/python`. Required unless you run the whole
  pipeline from inside that env.
- `--metal_rfdiffusion_script PATH` overrides the fork location (default
  `~/Code/RFdiffusion-mps/scripts/run_inference.py`).
- `--rfdiffusion_script PATH` takes precedence over both defaults for any
  backend, if you want to pin an exact checkout.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically so the few ops MPS does
  not implement transparently fall back to CPU.
- Stage 1 keeps the CUDA-OOM auto-downscaling loop; it now also recognises
  `MPS backend out of memory` and halves `inference.num_designs` on Metal.

## Fast validation run (no install needed)

`--dry-run` still skips all heavy models and writes schema-valid stubs so you
can exercise the full graph end-to-end:

```bash
cd retro_pipeline
source .venv/bin/activate

./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --dry-run \
  --backend mps \
  --protenix-runtime local \
  --top-n 10
```

(`--protenix-runtime local` is required with `--backend mps`: the dockerized
Protenix path is CUDA-only and is rejected up front even in `--dry-run`.)

## Optional mixed real-mode experiment

You can run selected stages manually:

```bash
# Stage 2 (ProteinMPNN) with MPS hint
python -m scripts.run_proteinmpnn \
  --config configs/sox2.yaml \
  --in_dir workspace/01_backbones \
  --out_dir workspace/02_sequences \
  --backend mps

# Stage 3 (Protenix) local runtime with MPS hint
python -m scripts.run_protenix \
  --config configs/sox2.yaml \
  --in_dir workspace/02_sequences \
  --out_dir workspace/03_predictions \
  --runtime local \
  --backend mps
```

If local Protenix fails on MPS in your environment, use `--backend cpu` or
switch stage 3 to cloud CUDA (`--resume-from ptx` on a GPU box).
