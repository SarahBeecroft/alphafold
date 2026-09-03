# Triton AlphaFold patch and container

This document describes the changes in the `amd_triton_patch` work and the
ROCm container added for running the patched AlphaFold implementation on AMD
GPUs.

## Summary

The patch adds an optional Triton implementation of the AlphaFold Evoformer
attention operation. The normal JAX implementation remains available as a
fallback. The container packages the patched source code together with the
ROCm/JAX/Triton runtime needed to execute that implementation.

## Source-code changes

### Triton attention kernel

[`alphafold/model/triton/evoformer_attn.py`](../alphafold/model/triton/evoformer_attn.py)
contains the Triton forward kernel and the JAX-to-Triton adapter. The adapter:

1. Converts AlphaFold tensors from `[batch, sequence, head, dimension]` into
   the kernel's layout.
2. Converts boolean attention masks into additive `0`/`-inf` masks.
3. Converts pair-bias tensors into the layout expected by the kernel.
4. Calls `jax_triton.triton_call`.
5. Converts the result back to AlphaFold's tensor layout.

### Optional dispatch

[`alphafold/model/triton/__init__.py`](../alphafold/model/triton/__init__.py)
detects whether Triton dependencies are installed and enables the path only
when `AF2_USE_TRITON=1`.

[`alphafold/model/modules.py`](../alphafold/model/modules.py) dispatches
compatible attention operations to Triton. The existing JAX implementation is
used when:

- Triton is not installed;
- `AF2_USE_TRITON` is not `1`;
- the operation is training rather than inference;
- key and value dimensions differ; or
- Triton is explicitly disabled for a model configuration.

Inference subbatching is bypassed for Triton-backed attention because the
kernel already performs memory-efficient attention. Template embedding uses a
scan instead of `vmap`, since the JAX-Triton integration does not provide a
vmap batching rule.

### OpenMM/JAX compatibility

[`alphafold/relax/amber_minimize.py`](../alphafold/relax/amber_minimize.py)
selects the HIP OpenMM platform for GPU relaxation on ROCm systems. Violation
metrics are evaluated on a CPU JAX device to avoid the GPU errors encountered
during that calculation.

### Tests and local setup

The Triton tests in [`tests/triton_tests/`](../tests/triton_tests/) cover:

- kernel output against a pure-JAX reference;
- integration with AlphaFold attention modules; and
- GPU/runtime smoke checks.

[`scripts/setup_triton_env.sh`](../scripts/setup_triton_env.sh) creates a
Python 3.12 virtual environment with the same JAX, ROCm plugin, Triton, and
`jax-triton` versions used by the container.

## Container implementation

[`docker/Dockerfile.triton`](../docker/Dockerfile.triton) is a separate image
from the existing NVIDIA/CUDA [`docker/Dockerfile`](../docker/Dockerfile).
It uses:

- `rocm/dev-ubuntu-24.04:7.1`;
- Python 3.12;
- JAX 0.8.2 and ROCm 7.1 JAX wheels;
- Triton 3.6.0;
- `jax-triton` 0.3.1;
- OpenMM 8.2.0;
- HH-suite 3.3.0, HMMER, and Kalign.

The Triton patch is included through:

```dockerfile
COPY . /app/alphafold
```

That command copies the complete patched checkout, including
`alphafold/model/triton/evoformer_attn.py` and the dispatch changes in
`alphafold/model/modules.py`. The Dockerfile then runs build-time checks that
fail if either of those patch components is missing.

The image defaults to `AF2_USE_TRITON=1`. Set it to `0` at runtime to use the
standard JAX attention implementation:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e AF2_USE_TRITON=1 \
  -v "$DOWNLOAD_DIR:/data:ro" \
  -v "$PWD:/work" \
  alphafold-triton \
  --fasta_paths=/work/your_protein.fasta \
  --data_dir=/data \
  --output_dir=/work/alphafold_output \
  --max_template_date=2022-01-01
```

## Build instructions

Run the build from the repository root. The final `.` is required because it
is the Docker build context containing the patched source:

```bash
docker build -f docker/Dockerfile.triton -t alphafold-triton .
```

Genetic databases and model parameters should be mounted at runtime rather
than copied into the image. This keeps the image manageable and avoids
embedding several hundred gigabytes of data in Docker layers.

## Validation

Python compilation and whitespace/diff checks passed in the development
environment. A full Docker build was not run there because Docker was not
installed. The first build on an ROCm-capable host should therefore be treated
as the final validation of the base image and ROCm wheel URLs.
