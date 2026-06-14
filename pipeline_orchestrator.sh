#!/usr/bin/env bash
# pipeline_orchestrator.sh — drive all four stages of the
# RetroSOX/RetroKLF in-silico replication pipeline end-to-end.
#
# Usage:
#   ./pipeline_orchestrator.sh \
#       --config configs/sox2.yaml \
#       [--workspace workspace/] \
#       [--dry-run] \
#       [--backend auto|cuda|mps|cpu] \
#       [--protenix-runtime docker|local] \
#       [--gpu-devices all|device=0] \
#       [--rfd-metal-script ~/Code/RFdiffusion-mps/scripts/run_inference.py] \
#       [--rfd-python ~/miniconda3/envs/RFdiffusion/bin/python] \
#       [--resume-from rfd|mpnn|ptx|foldx|rank] \
#       [--top-n 100]
#
# Each stage drops a `_DONE` sentinel in its output directory so a
# subsequent invocation with --resume-from will skip earlier successful
# stages cleanly.

set -euo pipefail

CONFIG=""
WORKSPACE="workspace"
DRY_RUN=""
RESUME_FROM="rfd"
TOP_N=""
BACKEND="auto"
PROTENIX_RUNTIME="docker"
GPU_DEVICES="all"
RFD_METAL_SCRIPT=""
RFD_PYTHON=""

usage() {
    cat <<'EOF'
Usage: pipeline_orchestrator.sh --config <yaml> [--workspace dir] [--dry-run]
                                [--backend auto|cuda|mps|cpu]
                                [--protenix-runtime docker|local]
                                [--gpu-devices all|device=0]
                                [--rfd-metal-script PATH]
                                [--rfd-python PATH]
                                [--resume-from rfd|mpnn|ptx|foldx|rank]
                                [--top-n N]

--rfd-metal-script: path to the Apple-Metal RFdiffusion fork's run_inference.py
                    (YaoYinYing/RFdiffusion@mps-test). Only used when --backend
                    is mps/cpu; defaults to ~/Code/RFdiffusion-mps/scripts/run_inference.py.
--rfd-python:       python interpreter that has RFdiffusion + its deps installed
                    (the fork lives in its own conda env), e.g.
                    ~/miniconda3/envs/RFdiffusion/bin/python.

Stages:
  rfd    RFdiffusion partial diffusion       -> workspace/01_backbones/
  mpnn   ProteinMPNN inverse folding         -> workspace/02_sequences/
  ptx    Protenix complex prediction+filter  -> workspace/03_predictions/
  foldx  FoldX BuildModel + Stability filter -> workspace/04_thermodynamics/
  rank   Composite ranking + top-N copy      -> workspace/final_top/
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2"; shift 2 ;;
        --workspace)     WORKSPACE="$2"; shift 2 ;;
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --backend)       BACKEND="$2"; shift 2 ;;
        --protenix-runtime) PROTENIX_RUNTIME="$2"; shift 2 ;;
        --gpu-devices)   GPU_DEVICES="$2"; shift 2 ;;
        --rfd-metal-script) RFD_METAL_SCRIPT="$2"; shift 2 ;;
        --rfd-python)    RFD_PYTHON="$2"; shift 2 ;;
        --resume-from)   RESUME_FROM="$2"; shift 2 ;;
        --top-n)         TOP_N="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config <yaml> is required." >&2
    usage
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config $CONFIG not found." >&2
    exit 2
fi

# Stable absolute paths so we can `cd` into the project root and still find things.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

mkdir -p "$WORKSPACE"/{01_backbones,02_sequences,03_predictions,04_thermodynamics,final_top,logs}

BACKBONES="$WORKSPACE/01_backbones"
SEQUENCES="$WORKSPACE/02_sequences"
PREDICTIONS="$WORKSPACE/03_predictions"
THERMO="$WORKSPACE/04_thermodynamics"
FINAL="$WORKSPACE/final_top"

PYTHON="${PYTHON:-python3}"

stage_done() {  # $1 = dir
    [[ -f "$1/_DONE" ]]
}

should_run() {  # $1 = stage_id, $2 = dir
    local stage="$1" dir="$2"
    case "$RESUME_FROM" in
        rfd)   return 0 ;;
        mpnn)  [[ "$stage" == "rfd" ]] && { stage_done "$dir" && return 1 || return 0; } ;;
        ptx)   [[ "$stage" == "rfd" || "$stage" == "mpnn" ]] && { stage_done "$dir" && return 1 || return 0; } ;;
        foldx) [[ "$stage" == "rfd" || "$stage" == "mpnn" || "$stage" == "ptx" ]] && { stage_done "$dir" && return 1 || return 0; } ;;
        rank)  [[ "$stage" != "rank" ]] && { stage_done "$dir" && return 1 || return 0; } ;;
    esac
    return 0
}

run_stage() {
    local label="$1"; shift
    echo "============================================================"
    echo " STAGE: $label"
    echo "============================================================"
    "$@"
}

# Stage 1: RFdiffusion ------------------------------------------------------
if should_run "rfd" "$BACKBONES"; then
    RFD_ARGS=(--config "$CONFIG" --out_dir "$BACKBONES" --backend "$BACKEND" --cuda_devices "$GPU_DEVICES")
    if [[ -n "$RFD_METAL_SCRIPT" ]]; then
        RFD_ARGS+=(--metal_rfdiffusion_script "$RFD_METAL_SCRIPT")
    fi
    if [[ -n "$RFD_PYTHON" ]]; then
        RFD_ARGS+=(--rfdiffusion_python "$RFD_PYTHON")
    fi
    run_stage "01 RFdiffusion (partial diffusion)" \
        "$PYTHON" -m scripts.run_rfdiffusion "${RFD_ARGS[@]}" $DRY_RUN
else
    echo "[skip] stage rfd already done ($BACKBONES/_DONE present)"
fi

# Stage 2: ProteinMPNN ------------------------------------------------------
if should_run "mpnn" "$SEQUENCES"; then
    run_stage "02 ProteinMPNN (fixed-motif inverse folding)" \
        "$PYTHON" -m scripts.run_proteinmpnn \
            --config "$CONFIG" \
            --in_dir "$BACKBONES" \
            --out_dir "$SEQUENCES" \
            --backend "$BACKEND" \
            --cuda_devices "$GPU_DEVICES" \
            $DRY_RUN
else
    echo "[skip] stage mpnn already done ($SEQUENCES/_DONE present)"
fi

# Stage 3: Protenix ---------------------------------------------------------
if should_run "ptx" "$PREDICTIONS"; then
    run_stage "03 Protenix (protein + dsDNA complex)" \
        "$PYTHON" -m scripts.run_protenix \
            --config "$CONFIG" \
            --in_dir "$SEQUENCES" \
            --out_dir "$PREDICTIONS" \
            --runtime "$PROTENIX_RUNTIME" \
            --backend "$BACKEND" \
            --gpu_devices "$GPU_DEVICES" \
            $DRY_RUN
else
    echo "[skip] stage ptx already done ($PREDICTIONS/_DONE present)"
fi

# Stage 4: FoldX ------------------------------------------------------------
if should_run "foldx" "$THERMO"; then
    run_stage "04 FoldX (BuildModel + Stability)" \
        "$PYTHON" -m scripts.run_foldx_filter \
            --config "$CONFIG" \
            --in_dir "$PREDICTIONS" \
            --sequences_dir "$SEQUENCES" \
            --out_dir "$THERMO" \
            $DRY_RUN
else
    echo "[skip] stage foldx already done ($THERMO/_DONE present)"
fi

# Stage 5: rank -------------------------------------------------------------
RANK_ARGS=(--predictions "$PREDICTIONS" --thermo "$THERMO" --out_dir "$FINAL")
if [[ -n "$TOP_N" ]]; then
    RANK_ARGS+=(--top_n "$TOP_N")
fi
run_stage "05 Composite ranking" "$PYTHON" -m scripts.score_and_rank "${RANK_ARGS[@]}"

echo "============================================================"
echo " DONE. Deliverables:"
echo "   $WORKSPACE/ranked_designs.csv"
echo "   $WORKSPACE/pareto_front.csv"
echo "   $FINAL/        (top-N predicted structures)"
echo "============================================================"
