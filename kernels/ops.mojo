"""GPU kernels for forward and backward passes of a transformer model.

All dimensions are runtime arguments. Each kernel is launched on the GPU
and uses shared-memory reductions, tiled matmul, or flat parallelism as
appropriate.
"""

# ── Imports ──────────────────────────────────────────────────────────────────

from std.math import ceildiv, sqrt, exp, log, tanh
from std.os.atomic import Atomic, Consistency
from std.sys.info import is_apple_gpu
from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.primitives import warp
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation


# ── Constants ────────────────────────────────────────────────────────────────

comptime THREADS_PER_BLOCK: UInt = 256

# Tile sizes for shared-memory matmul.
# BM x BK tile of X and BN x BK tile of W loaded per iteration.
# Each thread computes a SUB_M x SUB_N sub-tile of the output.
# Shared memory: 2 * TILE_K * max(TILE_M, TILE_N) * 4 bytes.
# TILE_M=TILE_N=32, TILE_K=8, SUB_M=SUB_N=4 -> 2*8*32*4 = 2 KB smem, 64 threads/block.
comptime TILE_M: Int = 64
comptime TILE_N: Int = 64
comptime TILE_K: Int = 16
comptime SUB_M: Int = 8
comptime SUB_N: Int = 8
comptime MM_THREADS: Int = (TILE_M // SUB_M) * (TILE_N // SUB_N)  # 64


# ── Embedding ────────────────────────────────────────────────────────────────


def embedding_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    indices: UnsafePointer[Int64, MutAnyOrigin],
    num_tokens: Int,
    embd_dim: Int,
):
    """output[i, d] = weight[indices[i], d]."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= num_tokens * embd_dim:
        return
    var token_idx = tid // embd_dim
    var dim_idx = tid % embd_dim
    var vocab_idx = Int(indices[token_idx])
    output[token_idx * embd_dim + dim_idx] = weight[vocab_idx * embd_dim + dim_idx]


# ── RMSNorm ──────────────────────────────────────────────────────────────────


def rmsnorm_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    rms_out: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    num_rows: Int,
    dim: Int,
):
    """y = x / sqrt(mean(x^2) + eps). One block per row."""
    var row = Int(block_idx.x)
    if row >= num_rows:
        return

    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()

    var row_offset = row * dim
    var ss: Float32 = 0.0
    for i in range(Int(tid), dim, Int(THREADS_PER_BLOCK)):
        var val = input[row_offset + i]
        ss += val * val
    smem[tid] = ss
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()

    if tid < UInt(WARP_SIZE):
        var warp_val: Float32 = smem[tid][0]
        warp_val = warp.sum(warp_val)
        if tid == 0:
            smem[0] = warp_val
    barrier()

    var mean_sq = smem[0][0] / Float32(dim)
    var rms = sqrt(mean_sq + 1e-5)
    if tid == 0:
        rms_out[row] = rms

    for i in range(Int(tid), dim, Int(THREADS_PER_BLOCK)):
        output[row_offset + i] = input[row_offset + i] / rms


# ── Linear (Tiled Matmul) ───────────────────────────────────────────────────


def linear_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """Y[m, n] = sum_k X[m, k] * W[n, k].

    output: [M, N], input: [M, K], weight: [N, K] (row-major).
    Tiled matmul with shared memory. Each block computes TILE_M x TILE_N output.
    """
    var x_smem = stack_allocation[
        TILE_M * TILE_K, Float32, address_space=AddressSpace.SHARED
    ]()
    var w_smem = stack_allocation[
        TILE_N * TILE_K, Float32, address_space=AddressSpace.SHARED
    ]()

    var block_m = Int(block_idx.y)
    var block_n = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var thread_m = (tid // (TILE_N // SUB_N)) * SUB_M
    var thread_n = (tid % (TILE_N // SUB_N)) * SUB_N

    var out_m = block_m * TILE_M + thread_m
    var out_n = block_n * TILE_N + thread_n

    var acc00: Float32 = 0; var acc01: Float32 = 0; var acc02: Float32 = 0; var acc03: Float32 = 0
    var acc10: Float32 = 0; var acc11: Float32 = 0; var acc12: Float32 = 0; var acc13: Float32 = 0
    var acc20: Float32 = 0; var acc21: Float32 = 0; var acc22: Float32 = 0; var acc23: Float32 = 0
    var acc30: Float32 = 0; var acc31: Float32 = 0; var acc32: Float32 = 0; var acc33: Float32 = 0

    var num_k_tiles = (K + TILE_K - 1) // TILE_K

    for k_tile in range(num_k_tiles):
        var k_base = k_tile * TILE_K

        for load_idx in range(tid, TILE_M * TILE_K, MM_THREADS):
            var lm = load_idx // TILE_K
            var lk = load_idx % TILE_K
            var gm = block_m * TILE_M + lm
            var gk = k_base + lk
            if gm < M and gk < K:
                x_smem[load_idx] = input[gm * K + gk]
            else:
                x_smem[load_idx] = 0.0

        for load_idx in range(tid, TILE_N * TILE_K, MM_THREADS):
            var ln = load_idx // TILE_K
            var lk = load_idx % TILE_K
            var gn = block_n * TILE_N + ln
            var gk = k_base + lk
            if gn < N and gk < K:
                w_smem[load_idx] = weight[gn * K + gk]
            else:
                w_smem[load_idx] = 0.0

        barrier()

        for kk in range(TILE_K):
            var x0 = x_smem[(thread_m + 0) * TILE_K + kk]
            var x1 = x_smem[(thread_m + 1) * TILE_K + kk]
            var x2 = x_smem[(thread_m + 2) * TILE_K + kk]
            var x3 = x_smem[(thread_m + 3) * TILE_K + kk]
            var w0 = w_smem[(thread_n + 0) * TILE_K + kk]
            var w1 = w_smem[(thread_n + 1) * TILE_K + kk]
            var w2 = w_smem[(thread_n + 2) * TILE_K + kk]
            var w3 = w_smem[(thread_n + 3) * TILE_K + kk]
            acc00 += x0 * w0; acc01 += x0 * w1; acc02 += x0 * w2; acc03 += x0 * w3
            acc10 += x1 * w0; acc11 += x1 * w1; acc12 += x1 * w2; acc13 += x1 * w3
            acc20 += x2 * w0; acc21 += x2 * w1; acc22 += x2 * w2; acc23 += x2 * w3
            acc30 += x3 * w0; acc31 += x3 * w1; acc32 += x3 * w2; acc33 += x3 * w3

        barrier()

    if out_m + 0 < M and out_n + 0 < N: output[(out_m + 0) * N + out_n + 0] = acc00
    if out_m + 0 < M and out_n + 1 < N: output[(out_m + 0) * N + out_n + 1] = acc01
    if out_m + 0 < M and out_n + 2 < N: output[(out_m + 0) * N + out_n + 2] = acc02
    if out_m + 0 < M and out_n + 3 < N: output[(out_m + 0) * N + out_n + 3] = acc03
    if out_m + 1 < M and out_n + 0 < N: output[(out_m + 1) * N + out_n + 0] = acc10
    if out_m + 1 < M and out_n + 1 < N: output[(out_m + 1) * N + out_n + 1] = acc11
    if out_m + 1 < M and out_n + 2 < N: output[(out_m + 1) * N + out_n + 2] = acc12
    if out_m + 1 < M and out_n + 3 < N: output[(out_m + 1) * N + out_n + 3] = acc13
    if out_m + 2 < M and out_n + 0 < N: output[(out_m + 2) * N + out_n + 0] = acc20
    if out_m + 2 < M and out_n + 1 < N: output[(out_m + 2) * N + out_n + 1] = acc21
    if out_m + 2 < M and out_n + 2 < N: output[(out_m + 2) * N + out_n + 2] = acc22
    if out_m + 2 < M and out_n + 3 < N: output[(out_m + 2) * N + out_n + 3] = acc23
    if out_m + 3 < M and out_n + 0 < N: output[(out_m + 3) * N + out_n + 0] = acc30
    if out_m + 3 < M and out_n + 1 < N: output[(out_m + 3) * N + out_n + 1] = acc31
    if out_m + 3 < M and out_n + 2 < N: output[(out_m + 3) * N + out_n + 2] = acc32
    if out_m + 3 < M and out_n + 3 < N: output[(out_m + 3) * N + out_n + 3] = acc33


# ── RoPE ─────────────────────────────────────────────────────────────────────


def rope_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    cos_buf: UnsafePointer[Float32, MutAnyOrigin],
    sin_buf: UnsafePointer[Float32, MutAnyOrigin],
    num_tokens: Int,
    num_heads: Int,
    head_dim: Int,
):
    """Apply rotary position embeddings to input."""
    var half_dim = head_dim // 2
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = num_tokens * num_heads * half_dim
    if tid >= total:
        return

    var token_head_idx = tid // half_dim
    var d = tid % half_dim
    var token_idx = token_head_idx // num_heads

    var base_offset = token_head_idx * head_dim
    var x1 = input[base_offset + d]
    var x2 = input[base_offset + half_dim + d]

    var cs_offset = token_idx * half_dim + d
    var c = cos_buf[cs_offset]
    var s = sin_buf[cs_offset]

    output[base_offset + d] = x1 * c + x2 * s
    output[base_offset + half_dim + d] = x1 * (-s) + x2 * c


# ── Flash Attention (Forward) ────────────────────────────────────────────────


def flash_attn_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    lse_out: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
    window_size: Int,
):
    """Flash Attention 2 forward with online softmax, O(T) memory.

    One thread per (b, qp, h). Each thread loops over all key positions,
    maintaining running softmax statistics (online softmax algorithm).

    output: [B*T*num_heads, head_dim]
    lse_out: [B*T*num_heads] -- log-sum-exp per query (saved for backward)
    q, k, v: [B*T*num_heads, head_dim]
    """
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = B * T * num_heads
    if tid >= total or head_dim > 128:
        return

    var bt = tid // num_heads
    var h = tid % num_heads
    var b = bt // T
    var qp = bt % T

    var q_base = tid * head_dim
    var scale = 1.0 / sqrt(Float32(head_dim))

    # Online softmax state: running max (m) and running sum-of-exp (l).
    # As each new key score arrives, rescale the accumulated output so the
    # softmax normalisation stays correct without materialising the full
    # T x T attention matrix.
    var m: Float32 = Float32.MIN
    var l: Float32 = 0.0

    var o_acc = stack_allocation[128, Float32, address_space=AddressSpace.LOCAL]()
    for d in range(head_dim):
        o_acc[d] = 0.0

    for kp in range(T):
        if kp > qp:
            break
        if window_size > 0 and kp < qp - window_size:
            continue

        var k_base = (b * T * num_heads + kp * num_heads + h) * head_dim
        var score: Float32 = 0.0
        for d in range(head_dim):
            score += q[q_base + d] * k[k_base + d]
        score *= scale

        # Online softmax update: adjust running max, rescale old accumulators
        var m_new = score if score > m else m
        var exp_old = exp(m - m_new)
        var exp_new = exp(score - m_new)
        var l_new = l * exp_old + exp_new

        var alpha = (l * exp_old) / l_new  # rescale factor for old O
        var beta = exp_new / l_new         # weight for new V
        var v_base = (b * T * num_heads + kp * num_heads + h) * head_dim
        for d in range(head_dim):
            o_acc[d] = alpha * o_acc[d] + beta * v[v_base + d]

        m = m_new
        l = l_new

    var out_base = tid * head_dim
    for d in range(head_dim):
        output[out_base + d] = o_acc[d]

    if l > 0.0:
        lse_out[tid] = m + log(l)
    else:
        lse_out[tid] = Float32.MIN


# ── Flash Attention (Backward) ───────────────────────────────────────────────


def flash_attn_bwd_precompute_d(
    d_out: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
):
    """D[bth] = sum_d O[bth,d] * dO[bth,d]."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= B * T * num_heads:
        return
    var base = tid * head_dim
    var acc: Float32 = 0.0
    for d in range(head_dim):
        acc += output[base + d] * grad_output[base + d]
    d_out[tid] = acc


def flash_attn_bwd_dq(
    grad_q: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    d_buf: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
    window_size: Int,
):
    """Compute dQ by recomputing attention weights on the fly.

    dQ[b,qp,h,d] = scale * sum_kp A[qp,kp] * (dA[qp,kp] - D[qp]) * K[kp,d]
    where A[qp,kp] = exp(Q[qp] . K[kp] * scale - lse[qp]).
    """
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = B * T * num_heads * head_dim
    if tid >= total:
        return

    var bth = tid // head_dim
    var d = tid % head_dim
    var bt = bth // num_heads
    var h = bth % num_heads
    var b = bt // T
    var qp = bt % T

    var q_base = bth * head_dim
    var scale = 1.0 / sqrt(Float32(head_dim))
    var lse_val = lse[bth]
    var d_val = d_buf[bth]

    var acc: Float32 = 0.0
    for kp in range(T):
        if kp > qp:
            break
        if window_size > 0 and kp < qp - window_size:
            continue

        var k_base = (b * T * num_heads + kp * num_heads + h) * head_dim

        var score: Float32 = 0.0
        for dd in range(head_dim):
            score += q[q_base + dd] * k[k_base + dd]
        score *= scale
        var a_val = exp(score - lse_val)

        # dA = sum_dd dO[qp,dd] * V[kp,dd]
        var v_base = (b * T * num_heads + kp * num_heads + h) * head_dim
        var do_base = bth * head_dim
        var da: Float32 = 0.0
        for dd in range(head_dim):
            da += grad_output[do_base + dd] * v[v_base + dd]

        # dS = A * (dA - D)
        var ds = a_val * (da - d_val)
        acc += ds * k[k_base + d]

    grad_q[tid] = acc * scale


def flash_attn_bwd_dkv(
    grad_k: UnsafePointer[Float32, MutAnyOrigin],
    grad_v: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[Float32, MutAnyOrigin],
    k: UnsafePointer[Float32, MutAnyOrigin],
    v: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    d_buf: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
    window_size: Int,
):
    """Compute dK and dV by looping over query positions, recomputing A on the fly."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = B * T * num_heads * head_dim
    if tid >= total:
        return

    var bth = tid // head_dim
    var d = tid % head_dim
    var bt = bth // num_heads
    var h = bth % num_heads
    var b = bt // T
    var kp = bt % T

    var k_base = bth * head_dim
    var scale = 1.0 / sqrt(Float32(head_dim))

    var dk_acc: Float32 = 0.0
    var dv_acc: Float32 = 0.0

    # Loop over query positions that attend to this key position:
    # qp >= kp (causal), qp <= kp + window_size (if windowed)
    var qp_start = kp
    var qp_end = T
    if window_size > 0 and kp + window_size < T:
        qp_end = kp + window_size + 1

    for qp in range(qp_start, qp_end):
        var q_base = (b * T * num_heads + qp * num_heads + h) * head_dim
        var lse_val = lse[b * T * num_heads + qp * num_heads + h]
        var d_val = d_buf[b * T * num_heads + qp * num_heads + h]

        var score: Float32 = 0.0
        for dd in range(head_dim):
            score += q[q_base + dd] * k[k_base + dd]
        score *= scale
        var a_val = exp(score - lse_val)

        var do_base = (b * T * num_heads + qp * num_heads + h) * head_dim
        dv_acc += a_val * grad_output[do_base + d]

        # dA for this (qp, kp)
        var v_base = bth * head_dim
        var da: Float32 = 0.0
        for dd in range(head_dim):
            da += grad_output[do_base + dd] * v[v_base + dd]

        # dS = A * (dA - D)
        var ds = a_val * (da - d_val)
        dk_acc += ds * q[q_base + d]

    grad_k[tid] = dk_acc * scale
    grad_v[tid] = dv_acc


# ── Elementwise ──────────────────────────────────────────────────────────────


def relu_squared_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """relu(x)^2."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var x = input[tid]
    var r = x if x > 0.0 else Float32(0.0)
    output[tid] = r * r


def softcap_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    cap: Float32,
    size: Int,
):
    """cap * tanh(x / cap)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = cap * tanh(input[tid] / cap)


# ── Value Embeddings ─────────────────────────────────────────────────────────


def ve_gate_fwd(
    gate_out: UnsafePointer[Float32, MutAnyOrigin],
    x_norm: UnsafePointer[Float32, MutAnyOrigin],
    gate_weight: UnsafePointer[Float32, MutAnyOrigin],
    num_tokens: Int,
    n_kv_head: Int,
    gate_channels: Int,
    embd_dim: Int,
):
    """gate[t, h] = 2 * sigmoid(sum_c W[h, c] * x[t, c]).

    gate_out: [num_tokens, n_kv_head]
    x_norm: [num_tokens, embd_dim] -- only first gate_channels used
    gate_weight: [n_kv_head, gate_channels]
    """
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= num_tokens * n_kv_head:
        return
    var t = tid // n_kv_head
    var h = tid % n_kv_head
    var acc: Float32 = 0.0
    for c in range(gate_channels):
        acc += gate_weight[h * gate_channels + c] * x_norm[t * embd_dim + c]
    # 2 * sigmoid(x) = 2 / (1 + exp(-x))
    var sig = 1.0 / (1.0 + exp(-acc))
    gate_out[tid] = 2.0 * sig


def ve_apply_fwd(
    v: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    ve: UnsafePointer[Float32, MutAnyOrigin],
    num_tokens: Int,
    n_kv_head: Int,
    head_dim: Int,
):
    """v[t, h, d] += gate[t, h] * ve[t, h, d] (in-place)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = num_tokens * n_kv_head * head_dim
    if tid >= total:
        return
    var th = tid // head_dim
    var d = tid % head_dim
    var g = gate[th]
    v[tid] += g * ve[tid]


# ── Residual Connections ─────────────────────────────────────────────────────


def add_residual_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    residual: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """output = residual + input."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = residual[tid] + input[tid]


def scaled_add_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    scale_a: UnsafePointer[Float32, MutAnyOrigin],
    scale_b: UnsafePointer[Float32, MutAnyOrigin],
    idx: Int,
    size: Int,
):
    """output = scale_a[idx] * a + scale_b[idx] * b (per-layer learned residual)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var sa = scale_a[idx]
    var sb = scale_b[idx]
    output[tid] = sa * a[tid] + sb * b[tid]


# ── Softmax ──────────────────────────────────────────────────────────────────


def softmax_fwd(
    output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    num_rows: Int,
    cols: Int,
):
    """Row-wise softmax. One block per row."""
    var row = Int(block_idx.x)
    if row >= num_rows:
        return

    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()
    var row_offset = row * cols

    # Max
    var m: Float32 = Float32.MIN
    for i in range(Int(tid), cols, Int(THREADS_PER_BLOCK)):
        var val = input[row_offset + i]
        if val > m:
            m = val
    smem[tid] = m
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            if smem[tid + active] > smem[tid]:
                smem[tid] = smem[tid + active]
        barrier()

    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.max(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var row_max = smem[0][0]

    # Exp and sum
    var s: Float32 = 0.0
    for i in range(Int(tid), cols, Int(THREADS_PER_BLOCK)):
        var val = exp(input[row_offset + i] - row_max)
        output[row_offset + i] = val
        s += val
    smem[tid] = s
    barrier()

    active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()

    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var row_sum = smem[0][0]

    # Normalize
    for i in range(Int(tid), cols, Int(THREADS_PER_BLOCK)):
        output[row_offset + i] = output[row_offset + i] / row_sum


# ── Cross-Entropy ────────────────────────────────────────────────────────────


def cross_entropy_fwd(
    loss_out: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    targets: UnsafePointer[Int64, MutAnyOrigin],
    num_tokens: Int,
    vocab_size: Int,
):
    """Per-token cross-entropy loss. One block per token."""
    var token = Int(block_idx.x)
    if token >= num_tokens:
        return

    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()
    var row_offset = token * vocab_size
    var target = Int(targets[token])

    # Max
    var m: Float32 = Float32.MIN
    for i in range(Int(tid), vocab_size, Int(THREADS_PER_BLOCK)):
        var val = logits[row_offset + i]
        if val > m:
            m = val
    smem[tid] = m
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            if smem[tid + active] > smem[tid]:
                smem[tid] = smem[tid + active]
        barrier()

    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.max(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var row_max = smem[0][0]

    # Sum exp
    var s: Float32 = 0.0
    for i in range(Int(tid), vocab_size, Int(THREADS_PER_BLOCK)):
        s += exp(logits[row_offset + i] - row_max)
    smem[tid] = s
    barrier()

    active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()

    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var log_sum_exp = row_max + log(smem[0][0])

    if tid == 0:
        if target >= 0:
            loss_out[token] = -logits[row_offset + target] + log_sum_exp
        else:
            loss_out[token] = 0.0


# ── Reduction ────────────────────────────────────────────────────────────────


def mean_reduce(
    output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Compute mean of input[0:size], store in output[0]. One block."""
    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()

    var s: Float32 = 0.0
    for i in range(Int(tid), size, Int(THREADS_PER_BLOCK)):
        s += input[i]
    smem[tid] = s
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()

    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            output[0] = wv / Float32(size)


# ── Backward: Cross-Entropy ──────────────────────────────────────────────────


def cross_entropy_softcap_bwd(
    grad_logits: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    targets: UnsafePointer[Int64, MutAnyOrigin],
    cap: Float32,
    num_tokens: Int,
    vocab_size: Int,
    loss_scale: Float32,
):
    """Fused backward: cross-entropy + softcap -> gradient w.r.t. capped logits.

    grad = (softmax(capped_logits) - one_hot) * loss_scale.
    One block per token.
    """
    var token = Int(block_idx.x)
    if token >= num_tokens:
        return

    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()
    var row_offset = token * vocab_size
    var target = Int(targets[token])

    if target < 0:
        for i in range(Int(tid), vocab_size, Int(THREADS_PER_BLOCK)):
            grad_logits[row_offset + i] = 0.0
        return

    # logits buffer already contains softcapped values from forward.
    # Compute softmax of capped logits.

    # Max
    var m: Float32 = Float32.MIN
    for i in range(Int(tid), vocab_size, Int(THREADS_PER_BLOCK)):
        var val = logits[row_offset + i]
        if val > m:
            m = val
    smem[tid] = m
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            if smem[tid + active] > smem[tid]:
                smem[tid] = smem[tid + active]
        barrier()
    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.max(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var row_max = smem[0][0]

    # Sum exp
    var s: Float32 = 0.0
    for i in range(Int(tid), vocab_size, Int(THREADS_PER_BLOCK)):
        s += exp(logits[row_offset + i] - row_max)
    smem[tid] = s
    barrier()

    active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()
    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var row_sum = smem[0][0]

    # grad = (softmax - one_hot) * scale
    # The softcap backward is already accounted for since we computed
    # softmax on the capped logits. The chain rule through softcap
    # would be needed if we wanted gradients w.r.t. pre-cap logits,
    # but for the lm_head backward we need grad w.r.t. capped logits.
    for i in range(Int(tid), vocab_size, Int(THREADS_PER_BLOCK)):
        var softmax_val = exp(logits[row_offset + i] - row_max) / row_sum
        if i == target:
            grad_logits[row_offset + i] = (softmax_val - 1.0) * loss_scale
        else:
            grad_logits[row_offset + i] = softmax_val * loss_scale


# ── Backward: Linear ────────────────────────────────────────────────────────


def linear_bwd_dx(
    grad_input: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """dX[m, k] = sum_n dY[m, n] * W[n, k].  (dX = dY @ W)

    grad_input: [M, K], grad_output: [M, N], weight: [N, K].
    Tiled: output tile [TILE_M, TILE_N] covers (m, k). Inner loop over N.
    """
    var a_smem = stack_allocation[TILE_M * TILE_K, Float32, address_space=AddressSpace.SHARED]()
    var b_smem = stack_allocation[TILE_K * TILE_N, Float32, address_space=AddressSpace.SHARED]()

    var block_m = Int(block_idx.y)
    var block_k = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var thread_m = (tid // (TILE_N // SUB_N)) * SUB_M
    var thread_k = (tid % (TILE_N // SUB_N)) * SUB_N
    var out_m = block_m * TILE_M + thread_m
    var out_k = block_k * TILE_N + thread_k

    var acc00: Float32 = 0; var acc01: Float32 = 0; var acc02: Float32 = 0; var acc03: Float32 = 0
    var acc10: Float32 = 0; var acc11: Float32 = 0; var acc12: Float32 = 0; var acc13: Float32 = 0
    var acc20: Float32 = 0; var acc21: Float32 = 0; var acc22: Float32 = 0; var acc23: Float32 = 0
    var acc30: Float32 = 0; var acc31: Float32 = 0; var acc32: Float32 = 0; var acc33: Float32 = 0

    var num_n_tiles = (N + TILE_K - 1) // TILE_K
    for n_tile in range(num_n_tiles):
        var n_base = n_tile * TILE_K

        # Load dY[m_range, n_chunk] into a_smem[TILE_M, TILE_K]
        for li in range(tid, TILE_M * TILE_K, MM_THREADS):
            var lm = li // TILE_K
            var ln = li % TILE_K
            var gm = block_m * TILE_M + lm
            var gn = n_base + ln
            a_smem[li] = grad_output[gm * N + gn] if gm < M and gn < N else Float32(0)

        # Load W[n_chunk, k_range] into b_smem[TILE_K, TILE_N]
        for li in range(tid, TILE_K * TILE_N, MM_THREADS):
            var ln = li // TILE_N
            var lk = li % TILE_N
            var gn = n_base + ln
            var gk = block_k * TILE_N + lk
            b_smem[li] = weight[gn * K + gk] if gn < N and gk < K else Float32(0)
        barrier()

        for nn in range(TILE_K):
            var a0 = a_smem[(thread_m+0)*TILE_K + nn]
            var a1 = a_smem[(thread_m+1)*TILE_K + nn]
            var a2 = a_smem[(thread_m+2)*TILE_K + nn]
            var a3 = a_smem[(thread_m+3)*TILE_K + nn]
            var b0 = b_smem[nn*TILE_N + (thread_k+0)]
            var b1 = b_smem[nn*TILE_N + (thread_k+1)]
            var b2 = b_smem[nn*TILE_N + (thread_k+2)]
            var b3 = b_smem[nn*TILE_N + (thread_k+3)]
            acc00 += a0*b0; acc01 += a0*b1; acc02 += a0*b2; acc03 += a0*b3
            acc10 += a1*b0; acc11 += a1*b1; acc12 += a1*b2; acc13 += a1*b3
            acc20 += a2*b0; acc21 += a2*b1; acc22 += a2*b2; acc23 += a2*b3
            acc30 += a3*b0; acc31 += a3*b1; acc32 += a3*b2; acc33 += a3*b3
        barrier()

    if out_m+0 < M and out_k+0 < K: grad_input[(out_m+0)*K + out_k+0] = acc00
    if out_m+0 < M and out_k+1 < K: grad_input[(out_m+0)*K + out_k+1] = acc01
    if out_m+0 < M and out_k+2 < K: grad_input[(out_m+0)*K + out_k+2] = acc02
    if out_m+0 < M and out_k+3 < K: grad_input[(out_m+0)*K + out_k+3] = acc03
    if out_m+1 < M and out_k+0 < K: grad_input[(out_m+1)*K + out_k+0] = acc10
    if out_m+1 < M and out_k+1 < K: grad_input[(out_m+1)*K + out_k+1] = acc11
    if out_m+1 < M and out_k+2 < K: grad_input[(out_m+1)*K + out_k+2] = acc12
    if out_m+1 < M and out_k+3 < K: grad_input[(out_m+1)*K + out_k+3] = acc13
    if out_m+2 < M and out_k+0 < K: grad_input[(out_m+2)*K + out_k+0] = acc20
    if out_m+2 < M and out_k+1 < K: grad_input[(out_m+2)*K + out_k+1] = acc21
    if out_m+2 < M and out_k+2 < K: grad_input[(out_m+2)*K + out_k+2] = acc22
    if out_m+2 < M and out_k+3 < K: grad_input[(out_m+2)*K + out_k+3] = acc23
    if out_m+3 < M and out_k+0 < K: grad_input[(out_m+3)*K + out_k+0] = acc30
    if out_m+3 < M and out_k+1 < K: grad_input[(out_m+3)*K + out_k+1] = acc31
    if out_m+3 < M and out_k+2 < K: grad_input[(out_m+3)*K + out_k+2] = acc32
    if out_m+3 < M and out_k+3 < K: grad_input[(out_m+3)*K + out_k+3] = acc33


def linear_bwd_dw(
    grad_weight: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """dW[n, k] += sum_m dY[m, n] * X[m, k].  (dW = dY^T @ X)

    grad_weight: [N, K], grad_output: [M, N], input: [M, K].
    Tiled: output tile [TILE_M, TILE_N] covers (n, k). Inner loop over M.
    """
    var a_smem = stack_allocation[TILE_M * TILE_K, Float32, address_space=AddressSpace.SHARED]()
    var b_smem = stack_allocation[TILE_K * TILE_N, Float32, address_space=AddressSpace.SHARED]()

    var block_n = Int(block_idx.y)
    var block_k = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var thread_n = (tid // (TILE_N // SUB_N)) * SUB_M
    var thread_k = (tid % (TILE_N // SUB_N)) * SUB_N
    var out_n = block_n * TILE_M + thread_n
    var out_k = block_k * TILE_N + thread_k

    var acc00: Float32 = 0; var acc01: Float32 = 0; var acc02: Float32 = 0; var acc03: Float32 = 0
    var acc10: Float32 = 0; var acc11: Float32 = 0; var acc12: Float32 = 0; var acc13: Float32 = 0
    var acc20: Float32 = 0; var acc21: Float32 = 0; var acc22: Float32 = 0; var acc23: Float32 = 0
    var acc30: Float32 = 0; var acc31: Float32 = 0; var acc32: Float32 = 0; var acc33: Float32 = 0

    var num_m_tiles = (M + TILE_K - 1) // TILE_K
    for m_tile in range(num_m_tiles):
        var m_base = m_tile * TILE_K

        # Load dY^T[n_range, m_chunk]: dY^T[n, m] = dY[m, n]
        for li in range(tid, TILE_M * TILE_K, MM_THREADS):
            var ln = li // TILE_K
            var lm = li % TILE_K
            var gn = block_n * TILE_M + ln
            var gm = m_base + lm
            a_smem[li] = grad_output[gm * N + gn] if gm < M and gn < N else Float32(0)

        # Load X[m_chunk, k_range]
        for li in range(tid, TILE_K * TILE_N, MM_THREADS):
            var lm = li // TILE_N
            var lk = li % TILE_N
            var gm = m_base + lm
            var gk = block_k * TILE_N + lk
            b_smem[li] = input[gm * K + gk] if gm < M and gk < K else Float32(0)
        barrier()

        for mm in range(TILE_K):
            var a0 = a_smem[(thread_n+0)*TILE_K + mm]
            var a1 = a_smem[(thread_n+1)*TILE_K + mm]
            var a2 = a_smem[(thread_n+2)*TILE_K + mm]
            var a3 = a_smem[(thread_n+3)*TILE_K + mm]
            var b0 = b_smem[mm*TILE_N + (thread_k+0)]
            var b1 = b_smem[mm*TILE_N + (thread_k+1)]
            var b2 = b_smem[mm*TILE_N + (thread_k+2)]
            var b3 = b_smem[mm*TILE_N + (thread_k+3)]
            acc00 += a0*b0; acc01 += a0*b1; acc02 += a0*b2; acc03 += a0*b3
            acc10 += a1*b0; acc11 += a1*b1; acc12 += a1*b2; acc13 += a1*b3
            acc20 += a2*b0; acc21 += a2*b1; acc22 += a2*b2; acc23 += a2*b3
            acc30 += a3*b0; acc31 += a3*b1; acc32 += a3*b2; acc33 += a3*b3
        barrier()

    if out_n+0 < N and out_k+0 < K: grad_weight[(out_n+0)*K + out_k+0] += acc00
    if out_n+0 < N and out_k+1 < K: grad_weight[(out_n+0)*K + out_k+1] += acc01
    if out_n+0 < N and out_k+2 < K: grad_weight[(out_n+0)*K + out_k+2] += acc02
    if out_n+0 < N and out_k+3 < K: grad_weight[(out_n+0)*K + out_k+3] += acc03
    if out_n+1 < N and out_k+0 < K: grad_weight[(out_n+1)*K + out_k+0] += acc10
    if out_n+1 < N and out_k+1 < K: grad_weight[(out_n+1)*K + out_k+1] += acc11
    if out_n+1 < N and out_k+2 < K: grad_weight[(out_n+1)*K + out_k+2] += acc12
    if out_n+1 < N and out_k+3 < K: grad_weight[(out_n+1)*K + out_k+3] += acc13
    if out_n+2 < N and out_k+0 < K: grad_weight[(out_n+2)*K + out_k+0] += acc20
    if out_n+2 < N and out_k+1 < K: grad_weight[(out_n+2)*K + out_k+1] += acc21
    if out_n+2 < N and out_k+2 < K: grad_weight[(out_n+2)*K + out_k+2] += acc22
    if out_n+2 < N and out_k+3 < K: grad_weight[(out_n+2)*K + out_k+3] += acc23
    if out_n+3 < N and out_k+0 < K: grad_weight[(out_n+3)*K + out_k+0] += acc30
    if out_n+3 < N and out_k+1 < K: grad_weight[(out_n+3)*K + out_k+1] += acc31
    if out_n+3 < N and out_k+2 < K: grad_weight[(out_n+3)*K + out_k+2] += acc32
    if out_n+3 < N and out_k+3 < K: grad_weight[(out_n+3)*K + out_k+3] += acc33


# ── Backward: RMSNorm ────────────────────────────────────────────────────────


def rmsnorm_bwd(
    grad_input: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    rms_saved: UnsafePointer[Float32, MutAnyOrigin],
    num_rows: Int,
    dim: Int,
):
    """RMSNorm backward. One block per row.

    dx = (dy - x * dot(dy, x) / (dim * rms^2)) / rms
    """
    var row = Int(block_idx.x)
    if row >= num_rows:
        return

    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()
    var row_offset = row * dim
    var rms = rms_saved[row]

    var dot: Float32 = 0.0
    for i in range(Int(tid), dim, Int(THREADS_PER_BLOCK)):
        dot += grad_output[row_offset + i] * input[row_offset + i]
    smem[tid] = dot
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()
    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            smem[0] = wv
    barrier()

    var dot_sum = smem[0][0]
    var rms_sq = rms * rms

    for i in range(Int(tid), dim, Int(THREADS_PER_BLOCK)):
        var x = input[row_offset + i]
        var dy = grad_output[row_offset + i]
        grad_input[row_offset + i] = (dy - x * dot_sum / (Float32(dim) * rms_sq)) / rms


# ── Backward: RoPE ───────────────────────────────────────────────────────────


def rope_bwd(
    grad_input: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    cos_buf: UnsafePointer[Float32, MutAnyOrigin],
    sin_buf: UnsafePointer[Float32, MutAnyOrigin],
    num_tokens: Int,
    num_heads: Int,
    head_dim: Int,
):
    """RoPE backward = forward with negated sin (inverse rotation)."""
    var half_dim = head_dim // 2
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total = num_tokens * num_heads * half_dim
    if tid >= total:
        return

    var token_head_idx = tid // half_dim
    var d = tid % half_dim
    var token_idx = token_head_idx // num_heads

    var base_offset = token_head_idx * head_dim
    var dy1 = grad_output[base_offset + d]
    var dy2 = grad_output[base_offset + half_dim + d]

    var cs_offset = token_idx * half_dim + d
    var c = cos_buf[cs_offset]
    var s = sin_buf[cs_offset]

    grad_input[base_offset + d] = dy1 * c - dy2 * s
    grad_input[base_offset + half_dim + d] = dy1 * s + dy2 * c


# ── Backward: Elementwise ───────────────────────────────────────────────────


def relu_squared_bwd(
    grad_input: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """d/dx relu(x)^2 = 2*x*dy if x > 0, else 0."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var x = input[tid]
    if x > 0.0:
        grad_input[tid] = 2.0 * x * grad_output[tid]
    else:
        grad_input[tid] = 0.0


# ── Backward: Embedding ─────────────────────────────────────────────────────


def embedding_bwd(
    grad_weight: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    indices: UnsafePointer[Int64, MutAnyOrigin],
    num_tokens: Int,
    embd_dim: Int,
):
    """grad_weight[indices[i], d] += grad_output[i, d] (atomic scatter-add)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= num_tokens * embd_dim:
        return
    var token_idx = tid // embd_dim
    var dim_idx = tid % embd_dim
    var vocab_idx = Int(indices[token_idx])
    comptime ord = Consistency.RELEASE if is_apple_gpu() else Consistency.SEQUENTIAL
    _ = Atomic.fetch_add[ordering=ord](
        grad_weight + vocab_idx * embd_dim + dim_idx,
        grad_output[token_idx * embd_dim + dim_idx],
    )


# ── Backward: Residual ──────────────────────────────────────────────────────


def scaled_add_bwd(
    grad_a: UnsafePointer[Float32, MutAnyOrigin],
    grad_b: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    scale_a: UnsafePointer[Float32, MutAnyOrigin],
    scale_b: UnsafePointer[Float32, MutAnyOrigin],
    idx: Int,
    size: Int,
):
    """Backward for output = sa*a + sb*b.

    grad_a = sa * grad_output, grad_b = sb * grad_output.
    (Scalar gradients computed separately via reduction.)
    """
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var sa = scale_a[idx]
    var sb = scale_b[idx]
    grad_a[tid] = sa * grad_output[tid]
    grad_b[tid] = sb * grad_output[tid]


def ve_apply_bwd(
    grad_ve: UnsafePointer[Float32, MutAnyOrigin],
    grad_gate: UnsafePointer[Float32, MutAnyOrigin],
    grad_v: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    ve: UnsafePointer[Float32, MutAnyOrigin],
    num_tokens: Int,
    n_kv_head: Int,
    head_dim: Int,
):
    """Backward for v += gate * ve.

    grad_ve[t,h,d] = gate[t,h] * grad_v[t,h,d]
    grad_gate[t,h] = sum_d ve[t,h,d] * grad_v[t,h,d]  (reduction over head_dim)

    One block per (t, h) for the grad_gate reduction.
    """
    var row = Int(block_idx.x)
    if row >= num_tokens * n_kv_head:
        return

    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()
    var base = row * head_dim
    var g = gate[row]

    var dg: Float32 = 0.0
    for d in range(Int(tid), head_dim, Int(THREADS_PER_BLOCK)):
        var gv = grad_v[base + d]
        grad_ve[base + d] = g * gv
        dg += ve[base + d] * gv
    smem[tid] = dg
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()
    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            grad_gate[row] = wv


def ve_gate_bwd(
    grad_x_norm_out: UnsafePointer[Float32, MutAnyOrigin],
    grad_gate_weight: UnsafePointer[Float32, MutAnyOrigin],
    grad_gate: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    x_norm: UnsafePointer[Float32, MutAnyOrigin],
    gate_weight: UnsafePointer[Float32, MutAnyOrigin],
    num_tokens: Int,
    n_kv_head: Int,
    gate_channels: Int,
    embd_dim: Int,
):
    """Backward for gate = 2*sigmoid(W @ x[:,:gate_channels]).

    grad_z = grad_gate * gate * (1 - gate/2)
    grad_W[h,c] += sum_t grad_z[t,h] * x[t,c]
    grad_x[t,c] += sum_h grad_z[t,h] * W[h,c]
    """
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= num_tokens * n_kv_head:
        return
    var t = tid // n_kv_head
    var h = tid % n_kv_head

    var g = gate[tid]
    var dz = grad_gate[tid] * g * (1.0 - g / 2.0)

    comptime ord = Consistency.RELEASE if is_apple_gpu() else Consistency.SEQUENTIAL
    for c in range(gate_channels):
        var x_val = x_norm[t * embd_dim + c]
        _ = Atomic.fetch_add[ordering=ord](
            grad_gate_weight + h * gate_channels + c, dz * x_val)
    for c in range(gate_channels):
        _ = Atomic.fetch_add[ordering=ord](
            grad_x_norm_out + t * embd_dim + c, dz * gate_weight[h * gate_channels + c])


def scalar_grad_reduce(
    grad_scalar: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    idx: Int,
    size: Int,
):
    """Reduce d(scalar)/d(loss) = sum(grad_output * input). One block.

    Accumulates into grad_scalar[idx] (e.g. for learned residual scalars).
    """
    var tid = thread_idx.x
    var smem = stack_allocation[Int(THREADS_PER_BLOCK), Float32, address_space=AddressSpace.SHARED]()

    var s: Float32 = 0.0
    for i in range(Int(tid), size, Int(THREADS_PER_BLOCK)):
        s += grad_output[i] * input[i]
    smem[tid] = s
    barrier()

    var active = THREADS_PER_BLOCK
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[tid + active]
        barrier()
    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            comptime ord = Consistency.RELEASE if is_apple_gpu() else Consistency.SEQUENTIAL
            _ = Atomic.fetch_add[ordering=ord](grad_scalar + idx, wv)


# ── Utility ──────────────────────────────────────────────────────────────────


def zero_buffer(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Zero out a device buffer."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    buf[tid] = 0.0
