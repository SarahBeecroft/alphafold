"""Integration tests: AF2 modules produce same output with/without Triton."""
import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from ml_collections import ConfigDict

from alphafold.model import modules
try:
    from alphafold.model.triton import TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")


def _run_triangle_attention(pair_act, pair_mask, params, rng, orientation='per_row'):
    """Run TriangleAttention with current AF2_USE_TRITON setting."""

    def forward(pair_act, pair_mask):
        c = ConfigDict({
            'num_head': 4,
            'gating': True,
            'orientation': orientation,
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

    fn = hk.transform(forward)
    if params is None:
        params = fn.init(rng, pair_act, pair_mask)
    output = fn.apply(params, rng, pair_act, pair_mask)
    return output, params


class TestTriangleAttentionIntegration:

    def _compare_with_and_without_triton(self, n_res=32, c_z=64, orientation='per_row'):
        rng = jax.random.PRNGKey(42)
        pair_act = jax.random.normal(rng, (n_res, n_res, c_z))
        pair_mask = jnp.ones((n_res, n_res))

        # Run without Triton
        os.environ.pop('AF2_USE_TRITON', None)
        import importlib
        from alphafold.model import modules, triton as triton_mod
        importlib.reload(triton_mod)
        importlib.reload(modules)
        out_jax, params = _run_triangle_attention(pair_act, pair_mask, None, rng, orientation)

        # Run with Triton
        os.environ['AF2_USE_TRITON'] = '1'
        importlib.reload(triton_mod)
        importlib.reload(modules)
        out_tri, _ = _run_triangle_attention(pair_act, pair_mask, params, rng, orientation)

        # Clean up
        os.environ.pop('AF2_USE_TRITON', None)
        importlib.reload(triton_mod)
        importlib.reload(modules)

        np.testing.assert_allclose(out_tri, out_jax, atol=1e-2, rtol=1e-2)

    def test_per_row(self):
        self._compare_with_and_without_triton(orientation='per_row')

    def test_per_column(self):
        self._compare_with_and_without_triton(orientation='per_column')

    def test_with_partial_mask(self):
        rng = jax.random.PRNGKey(42)
        n_res, c_z = 32, 64
        pair_act = jax.random.normal(rng, (n_res, n_res, c_z))
        pair_mask = jnp.ones((n_res, n_res))
        pair_mask = pair_mask.at[24:, :].set(0.0)
        pair_mask = pair_mask.at[:, 24:].set(0.0)

        os.environ.pop('AF2_USE_TRITON', None)
        import importlib
        from alphafold.model import modules, triton as triton_mod
        importlib.reload(triton_mod)
        importlib.reload(modules)
        out_jax, params = _run_triangle_attention(pair_act, pair_mask, None, rng)

        os.environ['AF2_USE_TRITON'] = '1'
        importlib.reload(triton_mod)
        importlib.reload(modules)
        out_tri, _ = _run_triangle_attention(pair_act, pair_mask, params, rng)

        os.environ.pop('AF2_USE_TRITON', None)
        importlib.reload(triton_mod)
        importlib.reload(modules)

        np.testing.assert_allclose(out_tri, out_jax, atol=1e-2, rtol=1e-2)
