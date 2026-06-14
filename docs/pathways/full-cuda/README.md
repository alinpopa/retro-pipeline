# Pathway: Full CUDA Execution (Production)

Use this mode on Linux with NVIDIA GPUs for real model execution.

## Requirements

- Ubuntu 22.04/24.04
- NVIDIA driver + CUDA runtime
- Docker + NVIDIA Container Toolkit
- RFdiffusion installed (`~/Code/RFdiffusion`, or pass `--rfdiffusion_script`)
- ProteinMPNN installed (`/opt/ProteinMPNN`)
- Protenix Docker image (`bytedance/protenix:latest`)
- FoldX binary on `PATH`

## Installing RFdiffusion (GPU-arch–aware)

The upstream `env/SE3nv.yml` pins ancient `pytorch=1.9 / cudatoolkit=11.1`,
which won't detect modern GPUs. Use the bundled setup scripts instead — they
pick a torch/DGL/CUDA combo that matches your card:

- **Ampere / Ada / Hopper** (A100, A6000, L40/L40S, RTX 3090/4090, H100 →
  sm_80/86/89/90): prebuilt wheels, fast.

```bash
bash setup/setup_rfdiffusion_cuda.sh
```

- **Blackwell** (RTX PRO 6000 Blackwell, RTX 5090, B200 → sm_120): needs
  CUDA 12.8 + torch ≥ 2.7 and a from-source DGL build (DGL ships no cu128
  wheels). Run on a CUDA-12.8 **devel** image:

```bash
bash setup/setup_rfdiffusion_blackwell.sh
```

Both scripts auto-detect the GPU's compute capability and refuse to run on the
wrong architecture. Each prints a final `torch.cuda.get_arch_list()` /
`get_device_name(0)` check so you can confirm the GPU is visible before
launching real jobs.

## Command

```bash
cd retro_pipeline
source .venv/bin/activate

./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --backend cuda \
  --protenix-runtime docker \
  --gpu-devices all
```

## Notes

- Use `--gpu-devices device=0` to pin a single GPU.
- Use `--resume-from <stage>` after interruption.
- Outputs:
  - `workspace/ranked_designs.csv`
  - `workspace/pareto_front.csv`
  - `workspace/final_top/*.cif`

