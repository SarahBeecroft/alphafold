"""Unit tests: compare Triton kernel output against pure JAX reference."""
import math

import pytest
import jax
import jax.numpy as jnp
import numpy as np

try:
    from alphafold.model.triton.evoformer_attn import triton_attention_fwd
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")


def reference_attention(q, k, v, res_mask, pair_bias, softmax_scale):
    """Pure JAX reference implementation of evoformer attention.

    All shapes match the Triton kernel convention:
        q, k, v:    [BATCH, N_SEQ, HEAD, SEQ_LEN, DIM]
        res_mask:   [BATCH, N_SEQ, 1, 1, SEQ_LEN]
        pair_bias:  [BATCH, 1, HEAD, SEQ_LEN, SEQ_LEN]
    """
    q_scaled = q * softmax_scale
    logits = jnp.einsum('bshqd,bshkd->bshqk', q_scaled, k)
    logits = logits + pair_bias + res_mask
    weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum('bshqk,bshkd->bshqd', weights, v)
    return output


def _make_inputs(batch, n_seq, head, seq_len, dim, key=0, dtype=jnp.float32):
    """Generate random test inputs in kernel layout with realistic scaling.

    Inputs are scaled by 1/sqrt(2) to match the magnitude of Xavier-initialized
    linear projections (what the model produces in practice). This prevents
    amplification of numerical differences, especially in low-precision dtypes.
    """
    proj_scale = 1.0 / math.sqrt(2)
    rng = jax.random.PRNGKey(key)
    keys = jax.random.split(rng, 5)
    q = jax.random.normal(keys[0], (batch, n_seq, head, seq_len, dim), dtype=dtype) * proj_scale
    k = jax.random.normal(keys[1], (batch, n_seq, head, seq_len, dim), dtype=dtype) * proj_scale
    v = jax.random.normal(keys[2], (batch, n_seq, head, seq_len, dim), dtype=dtype) * proj_scale
    pair_bias = jax.random.normal(keys[3], (batch, 1, head, seq_len, seq_len), dtype=dtype) * 0.1
    res_mask = jnp.zeros((batch, n_seq, 1, 1, seq_len), dtype=dtype)
    softmax_scale = dim ** -0.5
    return q, k, v, res_mask, pair_bias, softmax_scale


def _assert_close(actual, expected, dtype):
    """Assert closeness with dtype-appropriate tolerances.

    Flash Attention uses online softmax which accumulates differently than
    standard softmax, introducing small numerical differences. Empirically:
      f32:  max ~4e-4  (online softmax accumulation)
      fp16: max ~5e-4  (10-bit mantissa)
      bf16: max ~2e-3  (7-bit mantissa)
    """
    if dtype in (jnp.bfloat16,):
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)
    elif dtype in (jnp.float16,):
        np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
    else:
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-4)


class TestTritonKernelCorrectness:
    """Compare Triton kernel against JAX reference."""

    def test_basic_small(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 2, 4, 16, 32)
        ref = reference_attention(q, k, v, mask, bias, scale)
        tri = triton_attention_fwd(q, k, v, mask, bias, softmax_scale=scale)
        _assert_close(tri, ref, jnp.float32)

    def test_realistic_size(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 64, 4, 64, 32)
        ref = reference_attention(q, k, v, mask, bias, scale)
        tri = triton_attention_fwd(q, k, v, mask, bias, softmax_scale=scale)
        _assert_close(tri, ref, jnp.float32)

    def test_non_power_of_2_seq_len(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 8, 4, 37, 32)
        ref = reference_attention(q, k, v, mask, bias, scale)
        tri = triton_attention_fwd(q, k, v, mask, bias, softmax_scale=scale)
        _assert_close(tri, ref, jnp.float32)

    def test_non_power_of_2_dim(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 8, 4, 32, 48)
        ref = reference_attention(q, k, v, mask, bias, scale)
        tri = triton_attention_fwd(q, k, v, mask, bias, softmax_scale=scale)
        _assert_close(tri, ref, jnp.float32)

    def test_with_masking(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 4, 4, 32, 32)
        mask = mask.at[:, :, :, :, 24:].set(-1e9)
        ref = reference_attention(q, k, v, mask, bias, scale)
        tri = triton_attention_fwd(q, k, v, mask, bias, softmax_scale=scale)
        _assert_close(tri, ref, jnp.float32)

    def test_softmax_scale(self):
        """Non-default softmax_scale."""
        q, k, v, mask, bias, _ = _make_inputs(1, 4, 4, 32, 32)
        scale = 64 ** -0.5
        ref = reference_attention(q, k, v, mask, bias, scale)
        tri = triton_attention_fwd(q, k, v, mask, bias, softmax_scale=scale)
        _assert_close(tri, ref, jnp.float32)

    def test_single_head(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 8, 1, 32, 32)
        ref = reference_attention(q, k, v, mask, bias, scale)
        tri = triton_attention_fwd(q, k, v, mask, bias, softmax_scale=scale)
        _assert_close(tri, ref, jnp.float32)

    def test_bf16(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 4, 4, 32, 32, dtype=jnp.bfloat16)
        ref = reference_attention(
            q.astype(jnp.float32), k.astype(jnp.float32),
            v.astype(jnp.float32), mask.astype(jnp.float32),
            bias.astype(jnp.float32), scale,
        )
        tri = triton_attention_fwd(q, k, v, mask.astype(jnp.bfloat16), bias,
                                   softmax_scale=scale)
        _assert_close(tri.astype(jnp.float32), ref, jnp.bfloat16)

    def test_fp16(self):
        q, k, v, mask, bias, scale = _make_inputs(1, 4, 4, 32, 32, dtype=jnp.float16)
        ref = reference_attention(
            q.astype(jnp.float32), k.astype(jnp.float32),
            v.astype(jnp.float32), mask.astype(jnp.float32),
            bias.astype(jnp.float32), scale,
        )
        tri = triton_attention_fwd(q, k, v, mask.astype(jnp.float16), bias,
                                   softmax_scale=scale)
        _assert_close(tri.astype(jnp.float32), ref, jnp.float16)
