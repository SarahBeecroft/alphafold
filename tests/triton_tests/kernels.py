"""Triton kernel definitions for smoke tests.

Separated from the test file because @triton.jit uses inspect.getsource(),
which fails when pytest's assertion rewriter transforms the module bytecode.
"""
import triton
import triton.language as tl


@triton.jit
def _add_kernel(X, Y, O, N: tl.constexpr, BLOCK: tl.constexpr):
    """Trivial kernel: O = X + Y, element-wise."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(O + offs, x + y, mask=mask)


@triton.jit
def _softmax_kernel(
    X, O,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Simple row-wise softmax kernel — exercises pattern closer to attention."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf'))
    x_max = tl.max(x, axis=0)
    exp_x = tl.math.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    result = exp_x / sum_exp
    tl.store(O + row * stride_o + offs, result, mask=mask)
