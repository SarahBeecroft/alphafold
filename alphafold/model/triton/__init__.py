"""Triton kernel integration for AlphaFold-Multimer."""
import os

try:
    import jax_triton  # noqa: F401
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# User must opt in via environment variable
USE_TRITON = TRITON_AVAILABLE and os.environ.get('AF2_USE_TRITON', '') == '1'

if USE_TRITON:
    from alphafold.model.triton.evoformer_attn import triton_attention_af2
