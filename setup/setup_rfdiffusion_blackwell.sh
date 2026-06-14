#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# RFdiffusion setup for NVIDIA Blackwell GPUs (sm_120).
#
#   Examples: RTX PRO 6000 Blackwell, RTX 5090, B200 (compute capability 12.0).
#
# Why this script is different:
#   Blackwell (sm_120) needs CUDA 12.8 + PyTorch >= 2.7 (the cu128 wheels are
#   the only ones that ship sm_120 kernels). DGL has NO prebuilt wheels for
#   cu128 / torch 2.7+, so we BUILD DGL FROM SOURCE for sm_120.
#
# Requirements:
#   - a CUDA 12.8 *devel* image (nvcc that knows sm_120), e.g.
#     `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` on RunPod.
#   - build tools: git, cmake, ninja, a C++/CUDA toolchain (in the devel image).
#
# This is the heavy path and can take a while (DGL source build). If you are
# NOT locked to Blackwell, an Ampere/Ada/Hopper GPU + setup_rfdiffusion_cuda.sh
# is dramatically simpler.
#
# Override with env vars: RFD_DIR, RFD_REPO, DGL_DIR, DGL_REF, VENV_DIR,
#                         TORCH_INDEX, CUDA_ARCH
# ---------------------------------------------------------------------------
set -euo pipefail

RFD_DIR="${RFD_DIR:-$HOME/RFdiffusion}"
RFD_REPO="${RFD_REPO:-https://github.com/RosettaCommons/RFdiffusion}"
DGL_DIR="${DGL_DIR:-$HOME/dgl}"
DGL_REF="${DGL_REF:-}"                 # optional git tag/branch, e.g. v2.4.0
VENV_DIR="${VENV_DIR:-$HOME/se3nv}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
CUDA_ARCH="${CUDA_ARCH:-120}"          # sm_120 = Blackwell
NPROC="$(nproc 2>/dev/null || echo 4)"

echo "==> RFdiffusion Blackwell setup (sm_${CUDA_ARCH})"
echo "    RFD_DIR=$RFD_DIR  DGL_DIR=$DGL_DIR  VENV_DIR=$VENV_DIR  torch_index=$TORCH_INDEX"

# 0) Sanity checks ----------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,compute_cap --format=csv || {
    echo "WARN: nvidia-smi not found; cannot verify GPU arch."; }
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Use a CUDA 12.8 *devel* image (needed to build DGL for sm_120)."
    exit 1
fi
nvcc --version | sed -n 's/.*release \([0-9.]*\).*/nvcc CUDA \1/p' || true
command -v cmake >/dev/null 2>&1 || echo "WARN: cmake not found; will pip-install it into the venv."

# 1) Clone RFdiffusion -------------------------------------------------------
[ -d "$RFD_DIR" ] || git clone "$RFD_REPO" "$RFD_DIR"

# 2) Fresh venv + PyTorch (cu128, has sm_120) -------------------------------
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel cmake ninja
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"

# Verify the torch build actually contains sm_120 kernels before building DGL.
python - <<'PY'
import torch, sys
al = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| arch_list", al)
if not any(a.endswith("120") or a.endswith("100") for a in al):
    sys.exit("ERROR: this torch has no sm_120/sm_100 kernels. Install a cu128 torch>=2.7 "
             "(set TORCH_INDEX=https://download.pytorch.org/whl/cu128, or use nightly).")
PY

# 3) Build DGL from source for sm_120 ---------------------------------------
# DGL's runtime python deps (installed up front so `pip install -e .` is quick).
pip install torchdata==0.9.0 pydantic pandas scipy networkx tqdm psutil requests
if [ ! -d "$DGL_DIR" ]; then
    git clone --recurse-submodules https://github.com/dmlc/dgl.git "$DGL_DIR"
fi
cd "$DGL_DIR"
if [ -n "$DGL_REF" ]; then
    git checkout "$DGL_REF"
    git submodule update --init --recursive
fi
rm -rf build && mkdir -p build && cd build
# CMAKE_POLICY_VERSION_MINIMUM=3.5: DGL's vendored submodules (e.g. dmlc-core)
# still declare cmake_minimum_required < 3.5, which CMake 4.x rejects outright.
cmake -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DUSE_OPENMP=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j"$NPROC"
cd ../python && pip install -e .

# 4) RFdiffusion deps + SE3-Transformer (built for sm_120) + RFdiffusion ----
pip install hydra-core pyrsistent omegaconf icecream e3nn opt_einsum
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-cache-dir -r "$RFD_DIR/env/SE3Transformer/requirements.txt" || true
( cd "$RFD_DIR/env/SE3Transformer" && TORCH_CUDA_ARCH_LIST="12.0" python setup.py install )
pip install -e "$RFD_DIR"

# 5) Guard: deps above can clobber the cu128 torch — re-pin if so -----------
if ! python -c "import torch,sys; al=torch.cuda.get_arch_list(); sys.exit(0 if any(a.endswith('120') or a.endswith('100') for a in al) else 1)" >/dev/null 2>&1; then
    echo "==> torch lost its sm_120 kernels (clobbered by a dep); re-pinning cu128 wheel"
    pip install --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# 5b) torch>=2.6 (required for Blackwell) defaults torch.load(weights_only=True),
# which rejects RFdiffusion's trusted checkpoints AND e3nn's import-time
# constants.pt load. The global env-var override forces weights_only=False for
# every torch.load in the process (all entry points + third-party libs), which
# a per-file shim cannot do. Set it now and bake it into the venv activate.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if ! grep -q "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" "$VENV_DIR/bin/activate"; then
    echo 'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1' >> "$VENV_DIR/bin/activate"
    echo "==> Added TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 to $VENV_DIR/bin/activate"
fi

# 6) Verify ------------------------------------------------------------------
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
echo "    (activation also exports TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, required"
echo "     for torch>=2.6 to load RFdiffusion/e3nn checkpoints. If you run the"
echo "     bin/run_inference.py entry point WITHOUT sourcing activate, export it"
echo "     yourself: export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1)"
echo "    If 'no kernel image' STILL appears, your torch lacks sm_120 — switch to a"
echo "    cu128 torch>=2.7 (or nightly) and re-run, or use an Ampere/Ada/Hopper GPU."
