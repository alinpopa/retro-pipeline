# CLI Usage (Terminal)

This project is designed to run from terminal using either:

- the orchestrator: `./pipeline_orchestrator.sh`
- per-stage commands: `python -m scripts.<stage>`

All commands below assume:

```bash
cd retro_pipeline
source .venv/bin/activate
```

## 1) Orchestrator (recommended)

### Full pipeline

```bash
./pipeline_orchestrator.sh --config configs/sox2.yaml
```

### Small smoke run (fast)

```bash
./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --dry-run \
  --top-n 10
```

### Resume from a stage

```bash
./pipeline_orchestrator.sh --config configs/sox2.yaml --resume-from ptx
```

### Backend and runtime knobs

```bash
./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --backend cuda \
  --protenix-runtime docker \
  --gpu-devices all
```

Supported orchestrator flags:

- `--config`: target YAML (`configs/sox2.yaml` or `configs/klf4.yaml`)
- `--workspace`: output root (default `workspace`)
- `--dry-run`: generate stub outputs; no heavy model inference
- `--backend`: `auto|cuda|mps|cpu` (`mps`/`cpu` run RFdiffusion via the
  Apple-Metal fork — see the apple-metal pathway)
- `--protenix-runtime`: `docker|local`
- `--gpu-devices`: e.g. `all`, `0`, `0,1`, or `device=0`
- `--rfd-metal-script`: path to the Apple-Metal RFdiffusion fork's
  `run_inference.py` (default `~/Code/RFdiffusion-mps/scripts/run_inference.py`);
  only used when `--backend` is `mps`/`cpu`
- `--resume-from`: `rfd|mpnn|ptx|foldx|rank`
- `--top-n`: number of top structures copied to `workspace/final_top`

### Real diffusion on Apple Metal

```bash
./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --backend mps \
  --rfd-metal-script ~/Code/RFdiffusion-mps/scripts/run_inference.py \
  --protenix-runtime local
```

See `docs/pathways/apple-metal/README.md` for the one-time fork install.

## 2) Per-stage commands

### Stage 1: RFdiffusion

```bash
python -m scripts.run_rfdiffusion \
  --config configs/sox2.yaml \
  --out_dir workspace/01_backbones \
  --backend cuda \
  --cuda_devices all
```

Real diffusion on Apple Metal (uses the MPS fork):

```bash
python -m scripts.run_rfdiffusion \
  --config configs/sox2.yaml \
  --out_dir workspace/01_backbones \
  --backend mps \
  --num_designs 4 \
  --metal_rfdiffusion_script ~/Code/RFdiffusion-mps/scripts/run_inference.py
```

RFdiffusion stage flags:

- `--backend auto|cuda|mps|cpu` (`mps`/`cpu` route to the Metal fork)
- `--rfdiffusion_script PATH` pin an exact `run_inference.py` for any backend
- `--metal_rfdiffusion_script PATH` Metal-fork `run_inference.py` (default
  `~/Code/RFdiffusion-mps/scripts/run_inference.py`), used for `mps`/`cpu`
- `--cuda_devices all|0|0,1` GPU selector for `cuda`/`auto`

For quick testing (no install, stub outputs):

```bash
python -m scripts.run_rfdiffusion \
  --config configs/sox2.yaml \
  --out_dir workspace/01_backbones \
  --num_designs 4 \
  --dry-run
```

### Stage 2: ProteinMPNN

```bash
python -m scripts.run_proteinmpnn \
  --config configs/sox2.yaml \
  --in_dir workspace/01_backbones \
  --out_dir workspace/02_sequences \
  --backend cuda \
  --cuda_devices all
```

### Stage 3: Protenix

```bash
python -m scripts.run_protenix \
  --config configs/sox2.yaml \
  --in_dir workspace/02_sequences \
  --out_dir workspace/03_predictions \
  --runtime docker \
  --backend cuda \
  --gpu_devices all \
  --weights_cache "$HOME/.cache/protenix"
```

Apple/Metal attempt (local runtime only):

```bash
python -m scripts.run_protenix \
  --config configs/sox2.yaml \
  --in_dir workspace/02_sequences \
  --out_dir workspace/03_predictions \
  --runtime local \
  --backend mps
```

### Stage 4: FoldX filter

```bash
python -m scripts.run_foldx_filter \
  --config configs/sox2.yaml \
  --in_dir workspace/03_predictions \
  --sequences_dir workspace/02_sequences \
  --out_dir workspace/04_thermodynamics
```

### Stage 5: Rank and export top-N

```bash
python -m scripts.score_and_rank \
  --predictions workspace/03_predictions \
  --thermo workspace/04_thermodynamics \
  --out_dir workspace/final_top \
  --top_n 100
```

## 3) Practical presets

- **Linux + NVIDIA production**  
  `--backend cuda --protenix-runtime docker --gpu-devices all`
- **MacBook real diffusion on Metal**  
  `--backend mps --rfd-metal-script ~/Code/RFdiffusion-mps/scripts/run_inference.py`
  (stage 1 runs for real via the MPS fork; resume stage 3 on CUDA/cloud if
  local Protenix lacks Metal support)
- **MacBook validation only**  
  `--dry-run --backend mps` (stub outputs; no model install needed)
- **Hybrid local Protenix test**  
  `--runtime local --backend mps` (only if local Protenix build supports Metal in your environment)

