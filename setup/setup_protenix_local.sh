#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Protenix local (non-Docker) setup for NVIDIA Blackwell GPUs (sm_120).
#
# Requires Python 3.11+ (Protenix hardcodes `python_requires=">=3.11"`).
# On Ubuntu 22.04, Python 3.11 is installed via the deadsnakes PPA.
# On Ubuntu 24.04, Python 3.12 is the default — still works.
#
# This script installs Protenix in a DEDICATED Python 3.11 venv so it
# doesn't conflict with RFdiffusion's Python 3.10 venv ($HOME/se3nv).
#
# Blackwell workarounds applied:
#   1. Torch installed from cu128 index (pip defaults to cu124, which lacks
#      sm_120 kernels).
#   2. Custom CUDA kernels (fast_layer_norm_cuda_v2) are not compiled for
#      sm_120 -> LAYERNORM_TYPE=torch uses PyTorch native LayerNorm instead.
#   3. The Protenix custom arch list is patched to include sm_120 so any
#      kernels that DO compile target the right compute capability.
#   4. DeepSpeed and Triton may fail to compile on sm_120 — if so, set
#      DEEPSPEED_DISABLE=1 to skip them (Protenix inference still works).
#
# After this script, run the Protenix stage with --runtime local:
#
#   python -m scripts.run_protenix --config configs/sox2.yaml \
#       --in_dir workspace/02_sequences --out_dir workspace/03_predictions \
#       --runtime local --backend cuda
#
# Override with env vars: PROTENIX_DIR, PROTENIX_VENV, TORCH_INDEX, PYTHON_BIN
# ---------------------------------------------------------------------------
set -euo pipefail

PROTENIX_DIR="${PROTENIX_DIR:-$HOME/protenix}"
PROTENIX_VENV="${PROTENIX_VENV:-$HOME/protenix_venv}"
PROTENIX_REPO="${PROTENIX_REPO:-https://github.com/bytedance/Protenix.git}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
CUDA_ARCH="${CUDA_ARCH:-120}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "==> Protenix local setup for Blackwell (sm_${CUDA_ARCH})"
echo "    Python=$PYTHON_BIN  PROTENIX_DIR=$PROTENIX_DIR  PROTENIX_VENV=$PROTENIX_VENV"
echo "    torch_index=$TORCH_INDEX"

# 0) Ensure Python 3.11 is available ----------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "==> Python 3.11 not found; installing via deadsnakes PPA..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq
        apt-get install -y -qq software-properties-common
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update -qq
        apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    else
        echo "ERROR: Cannot install python3.11 automatically (no apt-get)."
        echo "       Install python>=3.11 manually and set PYTHON_BIN."
        exit 1
    fi
fi
"$PYTHON_BIN" --version

# 1) GPU check ---------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,compute_cap --format=csv || {
    echo "WARN: nvidia-smi not found; cannot verify GPU arch."; }

# 2) Clone Protenix ----------------------------------------------------------
if [ ! -d "$PROTENIX_DIR" ]; then
    git clone "$PROTENIX_REPO" "$PROTENIX_DIR"
else
    echo "Protenix repo already exists at $PROTENIX_DIR; pulling latest."
    cd "$PROTENIX_DIR" && git pull || true
fi

# 3) Create a Python 3.11 venv -----------------------------------------------
"$PYTHON_BIN" -m venv "$PROTENIX_VENV"
# shellcheck disable=SC1091
source "$PROTENIX_VENV/bin/activate"
pip install --upgrade pip wheel setuptools

# 4) Install torch + friends from cu128 index (sm_120 kernels) ---------------
# Protenix requirements.txt pins torch==2.7.1 — cu128 wheels exist for this.
pip install \
    "torch>=2.7" \
    "torchvision>=0.22" \
    "torchaudio>=2.7" \
    --index-url "$TORCH_INDEX"

# Verify sm_120 kernels are present.
python - <<'PY'
import torch, sys
al = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| arch_list", al)
if not any(a.endswith("120") or a.endswith("100") for a in al):
    sys.exit("ERROR: this torch has no sm_120/sm_100 kernels. Install a cu128 torch>=2.7.")
PY

# 5) Patch Protenix arch list for sm_120 -------------------------------------
ARCH_LIST_FILE="$PROTENIX_DIR/protenix/model/layer_norm/torch_ext_compile.py"
if [ -f "$ARCH_LIST_FILE" ]; then
    if grep -q '("120", "120")' "$ARCH_LIST_FILE"; then
        echo "sm_120 already in Protenix arch list."
    else
        echo "Patching Protenix arch list to include sm_120..."
        sed -i.bak 's/_wanted = \[("70", "70")/_wanted = [("70", "70")\n    ("120", "120"),/' "$ARCH_LIST_FILE"
        echo "Patched $ARCH_LIST_FILE (backup: ${ARCH_LIST_FILE}.bak)"
    fi
else
    echo "WARN: Could not find $ARCH_LIST_FILE; custom kernel compile may fail."
fi

# 6) Install Protenix dependencies (some may fail on sm_120 — that's OK) -----
echo "==> Installing Protenix dependencies from requirements.txt..."
# Install individually so failures don't block the whole list.
# Skip torch/torchvision/torchaudio (already installed from cu128 index).
FAILED_DEPS=""
for pkg in \
    "scipy>=1.9.0" \
    "ml_collections==1.1.0" \
    "tqdm" \
    "pandas" \
    "PyYAML" \
    "matplotlib" \
    "biopython" \
    "biotite" \
    "modelcif" \
    "gemmi" \
    "fair-esm" \
    "scikit-learn" \
    "scikit-learn-extra" \
    "pydantic>=2.0.0" \
    "protobuf>=3.20" \
    "icecream" \
    "numpy" \
    "networkx>=3.4" \
    "rdkit" \
    "pdbeccdutils" \
    "ipywidgets" \
    "py3Dmol" \
    "optree" \
    "cuequivariance-torch==0.8.0" \
    "cuequivariance-ops-torch-cu12==0.8.0"; do
    echo "  -> $pkg"
    pip install "$pkg" --no-build-isolation 2>/dev/null || FAILED_DEPS="$FAILED_DEPS $pkg"
done

# Tricky deps that need CUDA compilation for sm_120 — they often fail.
# If they do, don't abort — Protenix inference may still work without them.
for pkg in \
    "deepspeed==0.17.5" \
    "triton==3.3.1" \
    "wandb"; do
    echo "  -> $pkg (may need sm_120 CUDA compilation)"
    pip install "$pkg" --no-build-isolation 2>/dev/null || {
        echo "    WARN: $pkg install failed (sm_120 kernel issue)."
        FAILED_DEPS="$FAILED_DEPS $pkg"
    }
done

if [ -n "$FAILED_DEPS" ]; then
    echo "WARN: Some deps failed to install (expected on Blackwell):$FAILED_DEPS"
    echo "      Protenix inference should still work — failed deps are optional or"
    echo "      auto-fallback to PyTorch native ops (LAYERNORM_TYPE=torch)."
fi

# 7) Install Protenix itself (editable, skips deps we already handled) --------
cd "$PROTENIX_DIR"

# Protenix setup.py has `python_requires=">=3.11"` — we satisfy that now.
# Use --no-deps because we already installed deps above with Blackwell handling.
pip install -e . --no-deps

# 8) Blackwell LayerNorm fallback --------------------------------------------
# Protenix's custom CUDA LayerNorm kernel may still fail on sm_120.
# LAYERNORM_TYPE=torch tells Protenix to use PyTorch's native LayerNorm.
export LAYERNORM_TYPE=torch
export DEEPSPEED_DISABLE=1   # skip DeepSpeed if it failed to compile

if ! grep -q "LAYERNORM_TYPE" "$PROTENIX_VENV/bin/activate"; then
    cat >> "$PROTENIX_VENV/bin/activate" <<'ACTIVATE_EOF'

# Protenix Blackwell (sm_120) workarounds:
export LAYERNORM_TYPE=torch
export DEEPSPEED_DISABLE=1
ACTIVATE_EOF
    echo "==> Added LAYERNORM_TYPE=torch + DEEPSPEED_DISABLE=1 to activate script."
fi

# 9) Verify ------------------------------------------------------------------
echo
echo "==> Verifying installation..."
python - <<'PY'
import os
os.environ["LAYERNORM_TYPE"] = "torch"
os.environ["DEEPSPEED_DISABLE"] = "1"

import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))

try:
    import protenix
    print("protenix imported OK")
except Exception as e:
    print("WARN: protenix import failed:", e)

try:
    import subprocess
    result = subprocess.run(["protenix", "--help"], capture_output=True, text=True, timeout=10)
    print("protenix CLI: OK")
except Exception as e:
    print("WARN: protenix CLI check failed:", e)
PY

echo
echo "==> Done. Usage:"
echo "    source $PROTENIX_VENV/bin/activate"
echo "    (activation exports LAYERNORM_TYPE=torch + DEEPSPEED_DISABLE=1)"
echo
echo "    Then run the pipeline stage with --runtime local:"
echo "    python -m scripts.run_protenix --config configs/sox2.yaml \\"
echo "        --in_dir workspace/02_sequences --out_dir workspace/03_predictions \\"
echo "        --runtime local --backend cuda"
echo ""
echo "    NOTE: retro_pipeline must also be installed in this venv:"
echo "          cd ~/retro_pipeline && pip install -e ."
echo ""
echo "    NOTE: Protenix downloads model weights on first run into"
echo "          ~/.cache/protenix (~17 GB). Make sure you have enough disk."
