# Cloud Deployment Plan (Cost-Conscious)

This plan targets **RunPod** as the primary cloud due to lower on-demand GPU pricing and easy Docker workflows.

## Why RunPod

- Good spot pricing for A100 class GPUs
- Simple Docker-first workflow (matches Protenix/RFdiffusion usage)
- Easy switch between serverless and full pods

## Deployment goals

1. Keep costs low during iteration.
2. Support full CUDA production runs.
3. Preserve reproducibility and resumability.

## Architecture

```text
Laptop (authoring + launch)
    |
    | ssh / runpod API
    v
RunPod GPU Pod (A100 80GB preferred)
    - Docker + NVIDIA runtime
    - RFdiffusion + ProteinMPNN + Protenix + FoldX
    - workspace mounted on network volume
    - checkpoints and _DONE sentinels persisted
```

## Recommended rollout phases

### Phase 1: Validation (lowest cost)

- Use CPU-only local `--dry-run` to validate CLI contracts.
- In cloud, run tiny real tests:
  - `num_backbones=20`
  - `mpnn_seqs_per_backbone=5`
- Use a short-lived A40/L40S class pod if available.

Estimated cost (validation day): **$5–$20**

### Phase 2: Pilot production

- Use A100 80GB for realistic subsets:
  - `num_backbones=500`
  - `mpnn_seqs_per_backbone=20`
- Keep Protenix shard size conservative (32–64).
- Tune OOM behavior and throughput.

Estimated cost (pilot run): **$30–$120**

### Phase 3: Full run

- A100 80GB spot/on-demand depending queue pressure.
- `num_backbones=10000`, `mpnn_seqs_per_backbone=50`.
- Persist `workspace/` to volume; use `--resume-from` after interruptions.

Estimated cost (full run, one target): **$150–$700**  
(wide range depends on GPU availability, runtime efficiency, and retry rate)

## Cost controls

- Always start with `--dry-run` and small real subsets.
- Prefer spot instances where interruption tolerance is acceptable.
- Use `_DONE` + `--resume-from` to avoid recomputing finished stages.
- Keep Protenix `--max_jobs_per_shard` tuned to avoid expensive OOM retries.
- Auto-stop pods when queue is empty.
- Save only top artifacts:
  - `ranked_designs.csv`
  - `pareto_front.csv`
  - `final_top/*`

## Installing RFdiffusion on the pod (GPU-arch–aware)

RunPod hands you a wide range of GPUs, and the right RFdiffusion stack depends
on the card's compute capability. The upstream `env/SE3nv.yml`
(`pytorch=1.9 / cudatoolkit=11.1`) is too old to detect modern GPUs, so use the
bundled scripts:

| GPU class | compute cap | Image | Script |
|---|---|---|---|
| Ampere/Ada/Hopper (A100, L40S, A6000, 4090, H100) | 8.0–9.0 | CUDA 12.1 **devel** | `bash setup/setup_rfdiffusion_cuda.sh` |
| Blackwell (RTX PRO 6000 Blackwell, 5090, B200) | 12.0 | CUDA 12.8 **devel** | `bash setup/setup_rfdiffusion_blackwell.sh` |

Notes:
- Pick a **devel** image (nvcc is required to build SE3-Transformer, and DGL on
  Blackwell). `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` also ships conda.
- A Blackwell card needs a from-source DGL build (no cu128 wheels exist), which
  is slow and fragile — prefer an Ampere/Ada/Hopper pod unless you specifically
  need Blackwell.

## Suggested runbook

1. Launch RunPod with Ubuntu + CUDA **devel** image (12.1 for Ampere/Ada/Hopper,
   12.8 for Blackwell).
2. Clone repo and install the Python env; install RFdiffusion with the matching
   `setup/setup_rfdiffusion_*.sh` script above.
3. Mount/prepare persistent volume for `workspace`.
4. Pre-pull images and validate binaries.
5. Run:

```bash
./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --backend cuda \
  --protenix-runtime docker \
  --gpu-devices all
```

6. If interrupted:

```bash
./pipeline_orchestrator.sh --config configs/sox2.yaml --resume-from ptx
```

## Optional alternatives

- **Lambda Labs Cloud**: strong GPU offering, often slightly higher cost than RunPod.
- **Vast.ai**: can be cheaper, but more operational variance.
- **GCP/AWS**: best enterprise controls, typically highest cost for this workload.

For this project's cost target, RunPod is the best first choice.

