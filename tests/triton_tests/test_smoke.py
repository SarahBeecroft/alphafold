"""Smoke tests: can we call Triton kernels from JAX via jax-triton?"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

try:
    import jax_triton
    from tests.triton_tests.kernels import _add_kernel, _softmax_kernel
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="jax-triton not installed")


def test_trivial_triton_kernel():
    """GO/NO-GO: can jax-triton call a trivial kernel on this hardware?"""
    N = 128
    x = jnp.ones(N, dtype=jnp.float32)
    y = jnp.ones(N, dtype=jnp.float32) * 2.0

    out = jax_triton.triton_call(
        x, y,
        kernel=_add_kernel,
        out_shape=jax.ShapeDtypeStruct((N,), jnp.float32),
        grid=(1,),
        N=N,
        BLOCK=N,
    )

    np.testing.assert_allclose(out, np.full(N, 3.0), atol=1e-6)


def test_softmax_kernel():
    """Tests a kernel with strides and multiple program IDs."""
    ROWS, COLS = 4, 64
    x = jax.random.normal(jax.random.PRNGKey(0), (ROWS, COLS), dtype=jnp.float32)

    out = jax_triton.triton_call(
        x,
        kernel=_softmax_kernel,
        out_shape=jax.ShapeDtypeStruct((ROWS, COLS), jnp.float32),
        grid=(ROWS,),
        stride_x=COLS,
        stride_o=COLS,
        N=COLS,
        BLOCK=COLS,
    )

    expected = jax.nn.softmax(x, axis=-1)
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_gpu_available():
    """Verify JAX can see a GPU."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform in ('gpu', 'rocm', 'hip')]
    assert len(gpu_devices) > 0, (
        f"No GPU found. Available devices: {devices}. "
        f"Check ROCR_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES."
    )


def test_af2_modules_import():
    """Verify AF2 model modules work with this JAX version."""
    from alphafold.model import modules
    from alphafold.model import config
    from alphafold.model import common_modules
    # If any of these fail, JAX version incompatibility is a blocker.


def test_af2_attention_runs():
    """Verify AF2 Attention module produces output with the new JAX."""
    import haiku as hk
    from alphafold.model import modules
    from ml_collections import ConfigDict

    def forward(pair_act, pair_mask):
        c = ConfigDict({
            'num_head': 4,
            'gating': True,
            'orientation': 'per_row',
            'shared_dropout': True,
            'dropout_rate': 0.0,
        })
        gc = ConfigDict({
            'zero_init': True,
            'subbatch_size': 4,
            'deterministic': True,
        })
        attn = modules.TriangleAttention(c, gc)
        return attn(pair_act, pair_mask, is_training=False)

    N_RES, C_Z = 16, 64
    rng = jax.random.PRNGKey(42)
    pair_act = jax.random.normal(rng, (N_RES, N_RES, C_Z))
    pair_mask = jnp.ones((N_RES, N_RES))

    init_fn = hk.transform(forward)
    params = init_fn.init(rng, pair_act, pair_mask)
    output = init_fn.apply(params, rng, pair_act, pair_mask)

    assert output.shape == (N_RES, N_RES, C_Z)
    assert jnp.isfinite(output).all()
