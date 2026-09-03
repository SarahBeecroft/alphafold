"""Adapted Triton evoformer attention kernel for jax-triton.

Adapted from the AMD-authored forward-pass kernel contributed to OpenFold3.
Changes from the original:
1. Removed `import torch` and the `EvoformerAttention(torch.autograd.Function)` class
2. Reordered parameters: M and O moved after BATCH_SIZE (jax-triton appends
   output pointers after all positional args)
3. Stripped the `@triton.heuristics` decorator (EVEN_* computed in wrapper)
"""

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    res_mask_block_ptr,
    pair_bias_block_ptr,
    block_index_q,
    DIM,
    stride_K_seq,
    stride_V_seq,
    stride_mask_seq,
    stride_pair_bias_seq2,
    softmax_scale,
    EVEN_Q: tl.constexpr,
    EVEN_KV: tl.constexpr,
    EVEN_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    offs_d: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    """Run the inner loop of the forward pass of the attention mechanism."""
    lo, hi = 0, SEQ_LEN
    Q_block = Q_block * tl.full((1,), softmax_scale, dtype=Q_block.dtype)

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        if EVEN_Q & EVEN_KV:
            pair_bias_block = tl.load(pair_bias_block_ptr)
            res_mask_block = tl.load(res_mask_block_ptr).broadcast_to(
                (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
            )  # (1, BLOCK_SIZE_KV) -> (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
            if EVEN_DIM:
                K_block = tl.load(K_block_ptr)
                V_block = tl.load(V_block_ptr)
            else:
                K_block = tl.load(K_block_ptr, mask=offs_d[:, None] < DIM, other=0.0)
                V_block = tl.load(V_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
        else:
            pair_bias_block = tl.load(
                pair_bias_block_ptr,
                mask=(offs_q[:, None] < SEQ_LEN) & ((start_kv + offs_kv)[None, :] < SEQ_LEN),
                other=float("-inf"),
            )
            res_mask_block = tl.load(
                res_mask_block_ptr,
                mask=(start_kv + offs_kv)[None, :] < SEQ_LEN,
                other=float("-inf"),
            ).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
            if EVEN_DIM:
                K_block = tl.load(
                    K_block_ptr, mask=(start_kv + offs_kv)[None, :] < SEQ_LEN, other=0.0
                )
                V_block = tl.load(
                    V_block_ptr, mask=(start_kv + offs_kv)[:, None] < SEQ_LEN, other=0.0
                )
            else:
                K_block = tl.load(
                    K_block_ptr,
                    mask=((start_kv + offs_kv)[None, :] < SEQ_LEN) & (offs_d[:, None] < DIM),
                    other=0.0,
                )
                V_block = tl.load(
                    V_block_ptr,
                    mask=((start_kv + offs_kv)[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM),
                    other=0.0,
                )

        QK_block = tl.dot(Q_block, K_block) + pair_bias_block + res_mask_block

        # Need to mask out otherwise the softmax is wrong
        if not EVEN_KV:
            QK_block += tl.where((start_kv + offs_kv)[None, :] < SEQ_LEN, 0, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block = QK_block - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        P_block = P_block.to(V_block.dtype)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr += BLOCK_SIZE_KV * stride_V_seq
        K_block_ptr += BLOCK_SIZE_KV * stride_K_seq
        pair_bias_block_ptr += BLOCK_SIZE_KV * stride_pair_bias_seq2
        res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq

    return O_block, l_i, m_i


@triton.jit
def _attn_fwd(
    # Inputs (positional args from jax-triton)
    Q,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
    K,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
    V,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
    res_mask,  # BATCH_SIZE, N_SEQ, 1, 1, SEQ_LEN (accessed via strides)
    pair_bias,  # BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN
    softmax_scale,
    stride_Q_batch,
    stride_Q_msa,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_msa,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_msa,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_msa,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    stride_pair_bias_batch,
    stride_pair_bias_head,
    stride_pair_bias_seq1,
    stride_pair_bias_seq2,
    stride_mask_batch,
    stride_mask_msa,
    stride_mask_seq,
    BATCH_SIZE,
    # Outputs (appended by jax-triton after positional args)
    M,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN
    O,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
    # Constexprs (passed as metaparams by jax-triton)
    HEAD: tl.constexpr,
    N_SEQ: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    DIM: tl.constexpr,
    EVEN_Q: tl.constexpr,
    EVEN_KV: tl.constexpr,
    EVEN_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """Run the forward pass of the attention mechanism."""
    block_index_q = tl.program_id(0)

    index_batch_msa_head = tl.program_id(1)
    index_batch_msa = index_batch_msa_head // HEAD
    index_head = index_batch_msa_head % HEAD
    index_batch = index_batch_msa // N_SEQ
    index_msa = index_batch_msa % N_SEQ

    qvk_offset = (
        index_batch * stride_Q_batch + index_msa * stride_Q_msa + index_head * stride_Q_head
    )
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    offs_d = tl.arange(0, BLOCK_DIM)

    Q_block_ptr = Q + qvk_offset + (offs_q[:, None] * stride_Q_seq + offs_d[None, :])
    V_block_ptr = V + qvk_offset + (offs_kv[:, None] * stride_V_seq + offs_d[None, :])
    K_block_ptr = K + qvk_offset + (offs_kv[None, :] * stride_K_seq + offs_d[:, None])
    pair_bias_block_ptr = (
        pair_bias
        + index_batch * stride_pair_bias_batch
        + index_head * stride_pair_bias_head
        + (offs_q[:, None] * stride_pair_bias_seq1 + offs_kv[None, :] * stride_pair_bias_seq2)
    )
    O_block_ptr = O + qvk_offset + (offs_q[:, None] * stride_O_seq + offs_d[None, :])

    res_mask_block_ptr = (
        res_mask
        + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
        + (offs_kv[None, :] * stride_mask_seq)
    )

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    if EVEN_Q & EVEN_KV:
        if EVEN_DIM:
            Q_block = tl.load(Q_block_ptr)
        else:
            Q_block = tl.load(Q_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
    else:
        if EVEN_DIM:
            Q_block = tl.load(Q_block_ptr, mask=offs_q[:, None] < SEQ_LEN, other=0.0)
        else:
            Q_block = tl.load(
                Q_block_ptr, mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM), other=0.0
            )

    O_block, l_i, m_i = _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        res_mask_block_ptr,
        pair_bias_block_ptr,
        block_index_q,
        DIM,
        stride_K_seq,
        stride_V_seq,
        stride_mask_seq,
        stride_pair_bias_seq2,
        softmax_scale,
        EVEN_Q,
        EVEN_KV,
        EVEN_DIM,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        BLOCK_DIM,
        offs_q,
        offs_kv,
        offs_d,
        SEQ_LEN,
    )

    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    O_block = O_block.to(O.type.element_ty)
    m_ptrs = M + index_batch_msa_head * SEQ_LEN + offs_q

    if EVEN_Q:
        tl.store(m_ptrs, m_i)
        if EVEN_DIM:
            tl.store(O_block_ptr, O_block)
        else:
            tl.store(O_block_ptr, O_block, mask=offs_d[None, :] < DIM)
    else:
        tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
        if EVEN_DIM:
            tl.store(O_block_ptr, O_block, mask=offs_q[:, None] < SEQ_LEN)
        else:
            tl.store(
                O_block_ptr, O_block, mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM)
            )


def strides_from_shape(shape):
    """Compute element-count strides for a contiguous array."""
    strides = []
    product = 1
    for dim in reversed(shape):
        strides.append(product)
        product *= dim
    return list(reversed(strides))


def triton_attention_fwd(q, k, v, res_mask, pair_bias, softmax_scale=1.0):
    """Call the Triton evoformer attention kernel from JAX.

    Args:
        q: [BATCH, N_SEQ, HEAD, SEQ_LEN, DIM] query tensor
        k: [BATCH, N_SEQ, HEAD, SEQ_LEN, DIM] key tensor
        v: [BATCH, N_SEQ, HEAD, SEQ_LEN, DIM] value tensor
        res_mask: [BATCH, N_SEQ, 1, 1, SEQ_LEN] float mask (0=valid, -inf=masked)
        pair_bias: [BATCH, 1, HEAD, SEQ_LEN, SEQ_LEN] pair bias
        softmax_scale: scaling factor (default 1.0 if Q is pre-scaled)

    Returns:
        Output tensor [BATCH, N_SEQ, HEAD, SEQ_LEN, DIM]
    """
    import jax
    import jax.numpy as jnp
    import jax_triton

    BATCH, N_SEQ, HEAD, SEQ_LEN, DIM = q.shape
    BLOCK_DIM = max(triton.next_power_of_2(DIM), 16)

    # Compute heuristics manually (stripped from kernel decorator)
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_KV = 16
    EVEN_Q = SEQ_LEN % BLOCK_SIZE_Q == 0
    EVEN_KV = SEQ_LEN % BLOCK_SIZE_KV == 0
    EVEN_DIM = DIM == BLOCK_DIM

    # Compute strides (JAX arrays are always contiguous / row-major)
    q_strides = strides_from_shape(q.shape)
    k_strides = strides_from_shape(k.shape)
    v_strides = strides_from_shape(v.shape)
    o_strides = strides_from_shape(q.shape)  # O has same shape as Q
    pb_strides = strides_from_shape(pair_bias.shape)
    m_strides = strides_from_shape(res_mask.shape)

    grid = (
        triton.cdiv(SEQ_LEN, BLOCK_SIZE_Q),
        BATCH * N_SEQ * HEAD,
        1,
    )

    # Output shapes
    o_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    m_shape = jax.ShapeDtypeStruct(
        (BATCH, N_SEQ, HEAD, SEQ_LEN), jnp.float32
    )

    results = jax_triton.triton_call(
        # --- positional args (inputs + scalars) ---
        q, k, v, res_mask, pair_bias,
        softmax_scale,
        # Q strides
        q_strides[0], q_strides[1], q_strides[2], q_strides[3], q_strides[4],
        # K strides
        k_strides[0], k_strides[1], k_strides[2], k_strides[3], k_strides[4],
        # V strides
        v_strides[0], v_strides[1], v_strides[2], v_strides[3], v_strides[4],
        # O strides
        o_strides[0], o_strides[1], o_strides[2], o_strides[3], o_strides[4],
        # pair_bias strides (skip dim 1 which is broadcast)
        pb_strides[0], pb_strides[2], pb_strides[3], pb_strides[4],
        # mask strides (skip dims 2,3 which are 1)
        m_strides[0], m_strides[1], m_strides[4],
        # BATCH_SIZE
        BATCH,
        # --- outputs ---
        out_shape=[m_shape, o_shape],
        # --- kernel + grid ---
        kernel=_attn_fwd,
        grid=grid,
        # --- constexpr metaparams ---
        HEAD=HEAD,
        N_SEQ=N_SEQ,
        SEQ_LEN=SEQ_LEN,
        DIM=DIM,
        EVEN_Q=EVEN_Q,
        EVEN_KV=EVEN_KV,
        EVEN_DIM=EVEN_DIM,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_KV=BLOCK_SIZE_KV,
        BLOCK_DIM=BLOCK_DIM,
        num_warps=4,
        num_stages=1,
    )

    _M, O = results
    return O


def triton_attention_af2(q, k, v, mask, nonbatched_bias):
    """Triton attention with AF2 tensor conventions.

    Args:
        q: [batch_size, N_queries, num_head, key_dim] (already scaled by key_dim**-0.5)
        k: [batch_size, N_keys, num_head, key_dim]
        v: [batch_size, N_keys, num_head, value_dim]
        mask: [batch_size, 1, 1, N_keys] boolean mask (True=valid)
        nonbatched_bias: [num_head, N_queries, N_keys] or None

    Returns:
        [batch_size, N_queries, num_head, value_dim]
    """
    import jax.numpy as jnp

    assert q.shape[-1] == k.shape[-1] == v.shape[-1], (
        f"Triton kernel requires key_dim == value_dim, got "
        f"q={q.shape[-1]}, k={k.shape[-1]}, v={v.shape[-1]}"
    )

    batch, seq_len, num_head, dim = q.shape

    # Reshape AF2 [batch, seq, head, dim] -> Triton [1, batch, head, seq, dim]
    q_t = jnp.expand_dims(q, 0).transpose(0, 1, 3, 2, 4)
    k_t = jnp.expand_dims(k, 0).transpose(0, 1, 3, 2, 4)
    v_t = jnp.expand_dims(v, 0).transpose(0, 1, 3, 2, 4)

    # Convert boolean mask to float mask
    # AF2: [batch, 1, 1, N_keys] bool -> Triton: [1, batch, 1, 1, N_keys] float
    mask_float = jnp.where(mask, 0.0, -1e9).astype(q.dtype)
    mask_t = jnp.expand_dims(mask_float, 0)

    # Reshape bias
    # AF2: [head, N_q, N_k] -> Triton: [1, 1, head, N_q, N_k]
    if nonbatched_bias is not None:
        bias_t = nonbatched_bias[None, None, :, :, :].astype(q.dtype)
    else:
        bias_t = jnp.zeros((1, 1, num_head, seq_len, seq_len), dtype=q.dtype)

    # Call kernel (softmax_scale=1.0 because q is already scaled)
    o_t = triton_attention_fwd(q_t, k_t, v_t, mask_t, bias_t, softmax_scale=1.0)

    # Reshape back: Triton [1, batch, head, seq, dim] -> AF2 [batch, seq, head, dim]
    output = o_t[0].transpose(0, 2, 1, 3)

    return output
