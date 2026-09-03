#!/usr/bin/env bash
# Setup a venv for testing Triton triangle attention integration.
# Installs JAX 0.8.2 with ROCm 7.1 support, jax-triton, and AF2 deps.
# Usage: source scripts/setup_triton_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv/triton-attn"

if [[ -d "${VENV_DIR}" ]]; then
    echo "Venv already exists at ${VENV_DIR}"
    echo "To recreate, run: rm -rf ${VENV_DIR} && source $0"
    source "${VENV_DIR}/bin/activate"
    return 0 2>/dev/null || exit 0
fi

echo "Creating venv at ${VENV_DIR}..."
mkdir -p "$(dirname "${VENV_DIR}")"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

# JAX 0.8.2 with ROCm 7.1 support
# IMPORTANT: jax-triton 0.3.1 requires jax>=0.8.2. Using 0.8.0 results in a
# TritonKernel constructor mismatch that silently corrupts compiled kernels.
ROCM_JAX="https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.8.2"
pip install jax==0.8.2
pip install \
    "${ROCM_JAX}/jaxlib-0.8.2+rocm7-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
    "${ROCM_JAX}/jax_rocm7_pjrt-0.8.2+rocm7.1.1-py3-none-manylinux_2_28_x86_64.whl" \
    "${ROCM_JAX}/jax_rocm7_plugin-0.8.2+rocm7.1.1-cp312-cp312-manylinux_2_28_x86_64.whl"

# Triton 3.6.0 + jax-triton 0.3.1
pip install triton==3.6.0
pip install --no-deps jax-triton==0.3.1

# AF2 dependencies
pip install dm-haiku ml-collections absl-py numpy biopython

# Install alphafold in dev mode
pip install --no-deps -e "${REPO_ROOT}"

echo ""
echo "Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
