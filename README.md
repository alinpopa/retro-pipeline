# retro_pipeline — In silico replication of the Retro / OpenAI enhanced pioneer-TF workflow

This repository is a research-grade scaffold of the four-stage in silico
funnel reported by Retro Biosciences and OpenAI for generating
mutated variants of human reprogramming transcription factors (SOX2,
KLF4) that diverge ~26-36% in sequence from wild-type while drastically
improving pluripotency-induction efficiency.

The pipeline composes four well-known open-source tools:

1. **RFdiffusion (partial diffusion)** — generates 10,000 backbones with
   the native DNA-binding motif locked in 3D, the rest hallucinated.
2. **ProteinMPNN (fixed-motif inverse folding)** — assigns realistic
   amino-acid sequences to the new backbones while keeping the HMG-box
   (SOX2) or zinc-finger (KLF4) residues identical to wild-type.
3. **Protenix (AF3-class biomolecular complex prediction)** —
   ByteDance's open-source AlphaFold-3-equivalent. Verifies that the
   designed protein still binds its target dsDNA without spatial
   distortion. We use Protenix in place of AF3 because (a) its weights
   are freely available and (b) it natively supports protein + DNA +
   RNA + ligand complexes.
4. **FoldX (ΔΔG stability filter)** — discards designs that misfold or
   aggregate; ranks survivors by lowest hydrophobic surface exposure.

A composite Pareto ranker emits `ranked_designs.csv`, a `pareto_front.csv`,
and copies the top-N predicted complex structures into `workspace/final_top/`.

```text
templates/*.pdb
   │
   ▼
[1] RFdiffusion ──► workspace/01_backbones/*.pdb
   │
   ▼
[2] ProteinMPNN ──► workspace/02_sequences/*.fasta
   │
   ▼
[3] Protenix     ──► workspace/03_predictions/{*.cif, summary_confidences.json, metrics.csv}
   │
   ▼
[4] FoldX        ──► workspace/04_thermodynamics/ddg.csv
   │
   ▼
[5] score_and_rank ──► workspace/{ranked_designs.csv, pareto_front.csv}
                        workspace/final_top/rank0001__*.cif
```

## Target runtime

- Ubuntu 22.04 / 24.04 LTS
- 1× NVIDIA A100 / H100 (80 GB VRAM recommended; smaller cards may work
  with the built-in CUDA-OOM auto-downscaling)
- Docker + NVIDIA Container Toolkit
- Python ≥ 3.10

On a Mac (Apple Silicon, 24 GB RAM) you can run the **RFdiffusion diffusion
stage for real on Metal (`mps`)** via the drop-in MPS fork, plus ProteinMPNN
and the CPU stages; only the dockerized Protenix stage still needs CUDA. Every
stage also supports `--dry-run` (schema-valid placeholder outputs) so the
wiring is end-to-end testable without any model install. See
`docs/pathways/apple-metal/README.md`.

## Install

```bash
# 1. Python deps (the orchestrator and helpers).
cd retro_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. RFdiffusion (clone & install per upstream README).
git clone https://github.com/RosettaCommons/RFdiffusion ~/Code/RFdiffusion
# follow the upstream env_setup + model-weight download steps
#
# NVIDIA GPUs: the upstream env/SE3nv.yml pins pytorch=1.9/cuda11.1 and won't
# detect modern cards. Use the arch-aware helpers instead:
#   - Ampere/Ada/Hopper (A100/L40S/4090/H100): bash setup/setup_rfdiffusion_cuda.sh
#   - Blackwell (RTX PRO 6000 Blackwell/5090/B200):  bash setup/setup_rfdiffusion_blackwell.sh
# (see docs/pathways/full-cuda/README.md and docs/pathways/cloud-deployment/README.md)
#
# Apple Silicon: instead install the MPS-enabled fork into ~/Code/RFdiffusion-mps
# (git clone -b mps-test https://github.com/YaoYinYing/RFdiffusion ~/Code/RFdiffusion-mps)
# and follow docs/pathways/apple-metal/README.md. Then run --backend mps.

# 3. ProteinMPNN.
git clone https://github.com/dauparas/ProteinMPNN /opt/ProteinMPNN

# 4. Protenix (Docker).
docker pull bytedance/protenix:latest
# weights are auto-downloaded into a host-side cache the first run; bind it
# with `--weights_cache /path/on/host/.cache/protenix`.

# 5. FoldX (academic license required).
#    Download the Linux binary from https://foldxsuite.crg.eu/ and put
#    it on $PATH (or pass --foldx_bin /path/to/foldx).

# 6. Wild-type templates.
mkdir -p templates && cd templates
wget https://files.rcsb.org/download/6T7B.pdb   # SOX2-nucleosome (Dodonova 2020)
wget https://files.rcsb.org/download/6VTX.pdb   # KLF4 ZF + DNA (Sharma 2021)
```

## Usage

End-to-end on a real GPU box:

```bash
./pipeline_orchestrator.sh --config configs/sox2.yaml
```

Explicit backend/runtime control:

```bash
./pipeline_orchestrator.sh \
  --config configs/sox2.yaml \
  --backend cuda \
  --protenix-runtime docker \
  --gpu-devices all
```

End-to-end smoke test on a Mac (no real models needed):

```bash
./pipeline_orchestrator.sh --config configs/sox2.yaml --dry-run --top-n 10
```

Resume after a crash in the FoldX stage:

```bash
./pipeline_orchestrator.sh --config configs/sox2.yaml --resume-from foldx
```

## Execution pathways

- Pathway index: `docs/pathways/README.md`
- Full CUDA production: `docs/pathways/full-cuda/README.md`
- Apple Silicon / Metal: `docs/pathways/apple-metal/README.md`
- Dry-run and smoke testing: `docs/pathways/dry-run/README.md`
- Cost-aware cloud deployment (RunPod): `docs/pathways/cloud-deployment/README.md`
- CLI reference for terminal usage: `docs/cli/README.md`

Run a single stage manually:

```bash
python -m scripts.run_rfdiffusion --config configs/sox2.yaml \
    --out_dir workspace/01_backbones \
    --backend cuda --cuda_devices all
python -m scripts.run_proteinmpnn --config configs/sox2.yaml \
    --in_dir workspace/01_backbones --out_dir workspace/02_sequences \
    --backend cuda --cuda_devices all
python -m scripts.run_protenix --config configs/sox2.yaml \
    --in_dir workspace/02_sequences --out_dir workspace/03_predictions \
    --runtime docker --backend cuda --gpu_devices all \
    --weights_cache "$HOME/.cache/protenix"
python -m scripts.run_foldx_filter --config configs/sox2.yaml \
    --in_dir workspace/03_predictions --out_dir workspace/04_thermodynamics
python -m scripts.score_and_rank \
    --predictions workspace/03_predictions \
    --thermo workspace/04_thermodynamics \
    --top_n 100
```

## Configuration

Per-target YAML files live under `configs/`. The pipeline ships with
`sox2.yaml` (template `6T7B`, HMG-box residues 39-114, SOX2 consensus
motif) and `klf4.yaml` (template `6VTX`, ZF1-3 residues 430-512, NANOG DNA).
All knobs — partial-T noise, number of backbones, FoldX cutoff, etc. —
are exposed there.

Key knobs and their defaults (per the original spec):

| Field                  | Default | Meaning                                              |
| ---------------------- | ------- | ---------------------------------------------------- |
| `partial_T`            | 40      | RFdiffusion noise schedule. 35–45 → ~30% mutation space |
| `num_backbones`        | 10000   | Number of RFdiffusion samples                        |
| `mpnn_seqs_per_backbone` | 50    | ProteinMPNN sequences per backbone                   |
| `mpnn_temp`            | 0.1     | ProteinMPNN sampling temperature                     |
| `plddt_cutoff`         | 80      | Designed-region pLDDT cutoff                         |
| `interface_pae_cutoff` | 10.0    | Protein↔DNA chain-pair PAE (Å)                       |
| `ddg_cutoff`           | 0.0     | Positive ΔΔG → destabilizing → discard               |

## Note on the "iPAE > 10 Å" rule

The original spec mentions an `iPAE` metric, which is not a standard
field name. Protenix (and AF3) emit a `chain_pair_pae_min` matrix in
`summary_confidences.json`. Our filter computes the mean of all
off-diagonal entries of that matrix (i.e. the protein↔DNA inter-chain
PAE block) and applies the spec's 10 Å threshold to it. The
`interface_pae_cutoff` field in the YAML maps directly to this rule.

## CUDA OOM handling

Both RFdiffusion (stage 1) and Protenix (stage 3) are wrapped in
`common.run_with_oom_retry`. When the upstream subprocess emits any of
the standard CUDA-OOM strings (`CUDA out of memory`, `OutOfMemoryError`,
`cuBLAS_STATUS_ALLOC_FAILED`, …) — or the Metal equivalent (`MPS backend out
of memory`) — the wrapper halves the relevant batch size
(`inference.num_designs` for RFdiffusion; shard size for Protenix) and retries
up to 3×. Protenix additionally pre-shards its input JSON
(`--max_jobs_per_shard 64` by default) so a single job-list OOM doesn't
discard work that already succeeded in earlier shards.

## Backend/runtime parameters

The orchestrator and stage CLIs expose backend controls:

- `--backend auto|cuda|mps|cpu`
- `--gpu-devices` (GPU selector for CUDA-capable stages)
- `--protenix-runtime docker|local`

Current practical support:

- **RFdiffusion**: CUDA for production; **Apple Metal (`mps`) and `cpu` are
  supported for real runs** via the drop-in MPS fork
  (`YaoYinYing/RFdiffusion@mps-test`) — see `docs/pathways/apple-metal/README.md`.
  `--dry-run` still works with no install.
- **ProteinMPNN**: accepts backend hints (`mps/cpu` disable CUDA and enable fallback env)
- **Protenix**:
  - `runtime=docker` requires CUDA GPUs
  - `runtime=local` can be attempted with `mps/cpu` if your local Protenix build supports it
- **FoldX/Rank**: CPU-friendly

## Visualisation

The pipeline ships with a `retro-visualize` CLI and a Jupyter notebook
to explore every stage's output — metrics distributions, Pareto fronts,
the stage-by-stage survival funnel, and interactive 3D structure viewers.

### Install

```bash
pip install -e ".[viz]"           # from retro_pipeline/
# or: pip install 'retro-pipeline[viz]'
```

### TL;DR — one command for everything

```bash
retro-visualize report --open
```

Generates `workspace/viz_report.html` — a single self-contained page
with interactive Plotly charts for all stages plus py3Dmol 3D viewers
for backbones and passed predictions, and opens it in your browser.

### Per-stage plots (static PNG)

```bash
retro-visualize all              # everything below in one shot
retro-visualize funnel           # how many designs survive each stage
retro-visualize metrics          # Protenix confidence metrics (pLDDT, ipTM, PAE)
retro-visualize ddg              # FoldX ΔΔG distribution
retro-visualize ranking          # Pareto front + top-N composite ranking
```

Plots are saved to `workspace/` as numbered PNGs:
`00_pipeline_funnel.png`, `01_metrics_histograms.png`,
`02_metrics_scatter.png`, `03_ddg_distribution.png`,
`04_ranking_scatter.png`, `05_top_ranking.png`.

Pick a different format with `--format pdf|svg`.

### 3D structure viewers (standalone HTML)

```bash
retro-visualize backbones                        # overlay all backbone PDBs
retro-visualize structures                       # all passed predicted complexes
retro-visualize design sox2_00002__seq000 --open # one specific design by name
```

Each writes a standalone `.html` (uses py3Dmol/WebGL — no server needed).
Add `--open` to launch it in your default browser. Cap the number of
structures shown with `--max-structs 20`.

### Jupyter notebook

For deeper interactive exploration (sort/filter the ranked table,
re-run plots with custom parameters, browse any design by name):

```bash
jupyter notebook notebooks/pipeline_viz.ipynb
```

`jupyter` is included in the `[viz]` extra.

### Command reference

| Command | Reads from | Produces |
|---|---|---|
| `retro-visualize funnel` | every stage output | `00_pipeline_funnel.png` — attrition across the funnel |
| `retro-visualize metrics` | `03_predictions/metrics.csv` | pLDDT, ipTM, interface PAE histograms + scatter |
| `retro-visualize ddg` | `04_thermodynamics/ddg.csv` | ΔΔG histogram, ΔΔG vs hydrophobic SASA |
| `retro-visualize ranking` | `ranked_designs.csv`, `pareto_front.csv` | Pareto front, top-N composite score bars |
| `retro-visualize all` | all of the above | all PNGs at once |
| `retro-visualize report` | all of the above + `01_backbones/*.pdb`, `03_predictions/passed/*.cif` | one interactive HTML file |
| `retro-visualize backbones` | `01_backbones/*.pdb` | standalone 3D viewer |
| `retro-visualize structures` | `03_predictions/passed/*.cif` | standalone 3D viewer |
| `retro-visualize design <name>` | the named structure | standalone 3D viewer for a single design |

All commands accept `--workspace <path>` (default: `retro_pipeline/workspace`),
`--save <dir>` for plots (default: workspace), `--output <path>` for HTML,
and `--open` to launch in your browser.

## Citations

- Watson J. L. et al. *De novo design of protein structure and function
  with RFdiffusion.* Nature 620, 1089–1100 (2023).
- Dauparas J. et al. *Robust deep learning–based protein sequence design
  using ProteinMPNN.* Science 378, 49–56 (2022).
- ByteDance AML AI4Science. *Protenix: Toward High-Accuracy Open-Source
  Biomolecular Structure Prediction.* GitHub: bytedance/Protenix (2026).
- Schymkowitz J. et al. *The FoldX web server: an online force field.*
  Nucleic Acids Research 33, W382–W388 (2005).
- Dodonova S. O. et al. *Nucleosome-bound SOX2 and SOX11 structures
  elucidate pioneer factor function.* Nature 580, 669–672 (2020).

## Status

This is a scaffold — every CLI is wired and `--dry-run` exercises the
full graph end-to-end on a laptop. To go to production you still need
to: (a) download the upstream tool weights, (b) acquire a FoldX
academic license, (c) provide real `6T7B` / `6VTX` PDBs in
`templates/`.

Out of scope for v0 and tracked as easy follow-ups:

- MSA / template enrichment via `protenix prep`.
- Codon-optimization & Twist-style order CSV for wet-lab handoff.
- A `vendor/foldx/` helper that auto-downloads FoldX behind a license
  acceptance prompt.
