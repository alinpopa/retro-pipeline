#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RFdiffusion setup for NVIDIA Ampere / Ada / Hopper GPUs (sm_80/86/89/90).
#
#   Examples: A100, A6000, L40 / L40S, RTX 3090, RTX 4090, H100.
#
# Stack (all prebuilt wheels — no source builds):
#   - PyTorch 2.3.1 + cu121
#   - DGL 2.x (cu121 wheels)
#   - the SE3-Transformer bundled in RFdiffusion's env/SE3Transformer
#
# Use a CUDA 12.1 *devel* image (nvcc is needed to build SE3-Transformer),
# e.g. RunPod's CUDA-12.1 devel PyTorch template, or
# `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel` (which also ships conda).
#
# For Blackwell (sm_120, e.g. RTX PRO 6000 Blackwell / B200) use
# setup_rfdiffusion_blackwell.sh instead — cu121 has no sm_120 kernels.
#
# Override any of these with env vars before running:
#   RFD_DIR, RFD_REPO, VENV_DIR, TORCH_VERSION, TORCH_INDEX, DGL_FINDLINKS
# ---------------------------------------------------------------------------
set -euo pipefail

RFD_DIR="${RFD_DIR:-$HOME/RFdiffusion}"
RFD_REPO="${RFD_REPO:-https://github.com/RosettaCommons/RFdiffusion}"
VENV_DIR="${VENV_DIR:-$HOME/se3nv}"
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
DGL_FINDLINKS="${DGL_FINDLINKS:-https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html}"

echo "==> RFdiffusion CUDA setup (Ampere/Ada/Hopper)"
echo "    RFD_DIR=$RFD_DIR  VENV_DIR=$VENV_DIR  torch=$TORCH_VERSION ($TORCH_INDEX)"

# 0) Sanity checks ----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap --format=csv
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
    if [[ "$cc" == 12.* ]]; then
        echo "ERROR: detected compute capability $cc (Blackwell)."
        echo "       cu121 has no sm_120 kernels — run setup_rfdiffusion_blackwell.sh instead."
        exit 1
    fi
else
    echo "WARN: nvidia-smi not found; cannot verify GPU arch."
fi
command -v nvcc >/dev/null 2>&1 || echo "WARN: nvcc not found — use a *-devel CUDA image so SE3-Transformer can build."

# 1) Clone RFdiffusion -------------------------------------------------------
[ -d "$RFD_DIR" ] || git clone "$RFD_REPO" "$RFD_DIR"

# 2) Fresh venv (no conda required) -----------------------------------------
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

# 3) PyTorch for cu121 (runs on any 12.x driver) ----------------------------
pip install "torch==$TORCH_VERSION" torchvision torchaudio --index-url "$TORCH_INDEX"

# 4) DGL matching torch 2.3 + cu121 -----------------------------------------
pip install dgl -f "$DGL_FINDLINKS"

# 5) RFdiffusion's small deps -----------------------------------------------
pip install hydra-core pyrsistent omegaconf icecream e3nn opt_einsum

# 6) SE3-Transformer (bundled) + RFdiffusion --------------------------------
pip install --no-cache-dir -r "$RFD_DIR/env/SE3Transformer/requirements.txt"
( cd "$RFD_DIR/env/SE3Transformer" && python setup.py install )
pip install -e "$RFD_DIR"

# 7) Guard: deps above can clobber torch — re-pin the cu121 wheel if so ------
if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)" >/dev/null 2>&1; then
    echo "==> torch was overwritten with a non-CUDA build; re-pinning cu121 wheel"
    pip install --force-reinstall "torch==$TORCH_VERSION" torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# 8) Verify ------------------------------------------------------------------
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("arch_list", torch.cuda.get_arch_list())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
import dgl
print("dgl", dgl.__version__)
PY

echo
echo "==> Done. Activate with:  source $VENV_DIR/bin/activate"
echo "    Then run, e.g.:"
echo "    python $RFD_DIR/scripts/run_inference.py \\"
echo "      inference.input_pdb=6T7B.pdb \\"
echo "      'contigmap.contigs=[38-38/K39-114/203-203]' \\"
echo "      inference.num_designs=1 inference.output_prefix=01_backbones/sox2"
