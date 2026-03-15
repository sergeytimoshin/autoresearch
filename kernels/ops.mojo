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
# Each thread computes a SUB_M x SUB_N sub-tile of the output.
# Shared memory: 2 * TILE_K * max(TILE_M, TILE_N) * 4 bytes = 2 KB.
comptime TILE_M: Int = 32
comptime TILE_N: Int = 32
comptime TILE_K: Int = 8
comptime SUB_M: Int = 4
comptime SUB_N: Int = 4
comptime MM_THREADS: Int = (TILE_M // SUB_M) * (TILE_N // SUB_N)  # 64

# Flash Attention warp-parallel constants
comptime FA_WARPS: Int = 4        # warps per thread block
comptime FA_THREADS: Int = FA_WARPS * 32  # 128 threads per block


# ── Embedding ────────────────────────────────────────────────────────────────


def embedding_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    weight: UnsafePointer[BFloat16, MutAnyOrigin],
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
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    rms_out: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
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
        var val = Float32(input[row_offset + i])
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
        output[row_offset + i] = BFloat16(Float32(input[row_offset + i]) / rms)


# ── Linear (Tiled Matmul) ───────────────────────────────────────────────────


def linear_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
    weight: UnsafePointer[BFloat16, MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """Y[m, n] = sum_k X[m, k] * W[n, k].

    output: [M, N], input: [M, K], weight: [N, K] (row-major).
    Tiled matmul with shared memory. Load bf16, compute in f32.
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
                x_smem[load_idx] = Float32(input[gm * K + gk])
            else:
                x_smem[load_idx] = 0.0

        for load_idx in range(tid, TILE_N * TILE_K, MM_THREADS):
            var ln = load_idx // TILE_K
            var lk = load_idx % TILE_K
            var gn = block_n * TILE_N + ln
            var gk = k_base + lk
            if gn < N and gk < K:
                w_smem[load_idx] = Float32(weight[gn * K + gk])
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

    if out_m + 0 < M and out_n + 0 < N: output[(out_m + 0) * N + out_n + 0] = BFloat16(acc00)
    if out_m + 0 < M and out_n + 1 < N: output[(out_m + 0) * N + out_n + 1] = BFloat16(acc01)
    if out_m + 0 < M and out_n + 2 < N: output[(out_m + 0) * N + out_n + 2] = BFloat16(acc02)
    if out_m + 0 < M and out_n + 3 < N: output[(out_m + 0) * N + out_n + 3] = BFloat16(acc03)
    if out_m + 1 < M and out_n + 0 < N: output[(out_m + 1) * N + out_n + 0] = BFloat16(acc10)
    if out_m + 1 < M and out_n + 1 < N: output[(out_m + 1) * N + out_n + 1] = BFloat16(acc11)
    if out_m + 1 < M and out_n + 2 < N: output[(out_m + 1) * N + out_n + 2] = BFloat16(acc12)
    if out_m + 1 < M and out_n + 3 < N: output[(out_m + 1) * N + out_n + 3] = BFloat16(acc13)
    if out_m + 2 < M and out_n + 0 < N: output[(out_m + 2) * N + out_n + 0] = BFloat16(acc20)
    if out_m + 2 < M and out_n + 1 < N: output[(out_m + 2) * N + out_n + 1] = BFloat16(acc21)
    if out_m + 2 < M and out_n + 2 < N: output[(out_m + 2) * N + out_n + 2] = BFloat16(acc22)
    if out_m + 2 < M and out_n + 3 < N: output[(out_m + 2) * N + out_n + 3] = BFloat16(acc23)
    if out_m + 3 < M and out_n + 0 < N: output[(out_m + 3) * N + out_n + 0] = BFloat16(acc30)
    if out_m + 3 < M and out_n + 1 < N: output[(out_m + 3) * N + out_n + 1] = BFloat16(acc31)
    if out_m + 3 < M and out_n + 2 < N: output[(out_m + 3) * N + out_n + 2] = BFloat16(acc32)
    if out_m + 3 < M and out_n + 3 < N: output[(out_m + 3) * N + out_n + 3] = BFloat16(acc33)


def linear_fwd_f32(
    output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    """Y[m, n] = sum_k X[m, k] * W[n, k].

    Float32 variant for optimizer internal matmuls (Polar Express).
    output: [M, N], input: [M, K], weight: [N, K] (row-major).
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


def linear_bwd_dx_f32(
    grad_input: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    M: Int, N: Int, K: Int,
):
    """Float32 variant of linear_bwd_dx for optimizer internals."""
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
        for li in range(tid, TILE_M * TILE_K, MM_THREADS):
            var lm = li // TILE_K
            var ln = li % TILE_K
            var gm = block_m * TILE_M + lm
            var gn = n_base + ln
            a_smem[li] = grad_output[gm * N + gn] if gm < M and gn < N else Float32(0)
        for li in range(tid, TILE_K * TILE_N, MM_THREADS):
            var ln = li // TILE_N
            var lk = li % TILE_N
            var gn = n_base + ln
            var gk = block_k * TILE_N + lk
            b_smem[li] = weight[gn * K + gk] if gn < N and gk < K else Float32(0)
        barrier()
        for nn in range(TILE_K):
            var a0 = a_smem[(thread_m+0)*TILE_K + nn]; var a1 = a_smem[(thread_m+1)*TILE_K + nn]
            var a2 = a_smem[(thread_m+2)*TILE_K + nn]; var a3 = a_smem[(thread_m+3)*TILE_K + nn]
            var b0 = b_smem[nn*TILE_N + (thread_k+0)]; var b1 = b_smem[nn*TILE_N + (thread_k+1)]
            var b2 = b_smem[nn*TILE_N + (thread_k+2)]; var b3 = b_smem[nn*TILE_N + (thread_k+3)]
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


def linear_bwd_dw_f32(
    grad_weight: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    M: Int, N: Int, K: Int,
):
    """Float32 variant of linear_bwd_dw for optimizer internals."""
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
        for li in range(tid, TILE_M * TILE_K, MM_THREADS):
            var ln = li // TILE_K
            var lm = li % TILE_K
            var gn = block_n * TILE_M + ln
            var gm = m_base + lm
            a_smem[li] = grad_output[gm * N + gn] if gm < M and gn < N else Float32(0)
        for li in range(tid, TILE_K * TILE_N, MM_THREADS):
            var lm = li // TILE_N
            var lk = li % TILE_N
            var gm = m_base + lm
            var gk = block_k * TILE_N + lk
            b_smem[li] = input[gm * K + gk] if gm < M and gk < K else Float32(0)
        barrier()
        for mm in range(TILE_K):
            var a0 = a_smem[(thread_n+0)*TILE_K + mm]; var a1 = a_smem[(thread_n+1)*TILE_K + mm]
            var a2 = a_smem[(thread_n+2)*TILE_K + mm]; var a3 = a_smem[(thread_n+3)*TILE_K + mm]
            var b0 = b_smem[mm*TILE_N + (thread_k+0)]; var b1 = b_smem[mm*TILE_N + (thread_k+1)]
            var b2 = b_smem[mm*TILE_N + (thread_k+2)]; var b3 = b_smem[mm*TILE_N + (thread_k+3)]
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


# ── RoPE ─────────────────────────────────────────────────────────────────────


def rope_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
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
    var x1 = Float32(input[base_offset + d])
    var x2 = Float32(input[base_offset + half_dim + d])

    var cs_offset = token_idx * half_dim + d
    var c = cos_buf[cs_offset]
    var s = sin_buf[cs_offset]

    output[base_offset + d] = BFloat16(x1 * c + x2 * s)
    output[base_offset + half_dim + d] = BFloat16(x1 * (-s) + x2 * c)


# ── Flash Attention (Forward) ────────────────────────────────────────────────


def flash_attn_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    lse_out: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[BFloat16, MutAnyOrigin],
    k: UnsafePointer[BFloat16, MutAnyOrigin],
    v: UnsafePointer[BFloat16, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
    window_size: Int,
):
    """Warp-parallel Flash Attention 2 forward with online softmax.

    One warp (32 threads) per query. Each thread handles head_dim/32 elements
    of the dot product and output, using warp reductions for the full dot.

    Grid: ceil(B*T*H / FA_WARPS),  Block: FA_THREADS (128)
    output: [B*T*num_heads, head_dim]
    lse_out: [B*T*num_heads]
    q, k, v: [B*T*num_heads, head_dim]
    """
    var ws = Int(WARP_SIZE)
    var lane = Int(thread_idx.x) % ws
    var warp_id = Int(thread_idx.x) // ws
    var warps_in_block = Int(block_dim.x) // ws
    var query_idx = Int(block_idx.x) * warps_in_block + warp_id
    var total = B * T * num_heads
    if query_idx >= total:
        return

    var bt = query_idx // num_heads
    var h = query_idx % num_heads
    var b = bt // T
    var qp = bt % T

    var q_base = query_idx * head_dim
    var scale = 1.0 / sqrt(Float32(head_dim))
    var ept = head_dim // ws  # elements per thread (4 for D=128, W=32)
    var my_start = lane * ept

    # Load Q into registers (bf16 → f32)
    var q_reg = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    for e in range(ept):
        q_reg[e] = Float32(q[q_base + my_start + e])

    # Online softmax state
    var m: Float32 = Float32.MIN
    var l: Float32 = 0.0

    # Output accumulator in registers (f32)
    var o_reg = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    for e in range(ept):
        o_reg[e] = 0.0

    var kp_start = 0
    if window_size > 0 and qp > window_size:
        kp_start = qp - window_size
    for kp in range(kp_start, qp + 1):
        var k_base = (b * T * num_heads + kp * num_heads + h) * head_dim

        # Parallel dot product: each thread computes ept partial products
        var partial: Float32 = 0.0
        for e in range(ept):
            partial += q_reg[e] * Float32(k[k_base + my_start + e])
        var score = warp.sum(partial) * scale

        # Online softmax update
        var m_new = score if score > m else m
        var exp_old = exp(m - m_new)
        var exp_new = exp(score - m_new)
        var l_new = l * exp_old + exp_new

        var alpha = (l * exp_old) / l_new
        var beta = exp_new / l_new
        var v_base = (b * T * num_heads + kp * num_heads + h) * head_dim
        for e in range(ept):
            o_reg[e] = alpha * o_reg[e] + beta * Float32(v[v_base + my_start + e])

        m = m_new
        l = l_new

    # Store output (f32 → bf16)
    var out_base = query_idx * head_dim
    for e in range(ept):
        output[out_base + my_start + e] = BFloat16(o_reg[e])

    if lane == 0:
        if l > 0.0:
            lse_out[query_idx] = m + log(l)
        else:
            lse_out[query_idx] = Float32.MIN


# ── Flash Attention (Backward) ───────────────────────────────────────────────


def flash_attn_bwd_precompute_d(
    d_out: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
):
    """D[bth] = sum_d O[bth,d] * dO[bth,d]. Warp-parallel."""
    var ws = Int(WARP_SIZE)
    var lane = Int(thread_idx.x) % ws
    var warp_id = Int(thread_idx.x) // ws
    var warps_in_block = Int(block_dim.x) // ws
    var query_idx = Int(block_idx.x) * warps_in_block + warp_id
    if query_idx >= B * T * num_heads:
        return
    var base = query_idx * head_dim
    var ept = head_dim // ws
    var my_start = lane * ept
    var partial: Float32 = 0.0
    for e in range(ept):
        partial += Float32(output[base + my_start + e]) * grad_output[base + my_start + e]
    var result = warp.sum(partial)
    if lane == 0:
        d_out[query_idx] = result


def flash_attn_bwd_dq(
    grad_q: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[BFloat16, MutAnyOrigin],
    k: UnsafePointer[BFloat16, MutAnyOrigin],
    v: UnsafePointer[BFloat16, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    d_buf: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
    window_size: Int,
):
    """Warp-parallel dQ: one warp per query, recomputing attention on the fly."""
    var ws = Int(WARP_SIZE)
    var lane = Int(thread_idx.x) % ws
    var warp_id = Int(thread_idx.x) // ws
    var warps_in_block = Int(block_dim.x) // ws
    var query_idx = Int(block_idx.x) * warps_in_block + warp_id
    var total = B * T * num_heads
    if query_idx >= total:
        return

    var bt = query_idx // num_heads
    var h = query_idx % num_heads
    var b = bt // T
    var qp = bt % T

    var q_base = query_idx * head_dim
    var scale = 1.0 / sqrt(Float32(head_dim))
    var lse_val = lse[query_idx]
    var d_val = d_buf[query_idx]
    var ept = head_dim // ws
    var my_start = lane * ept

    # Load Q and dO into registers
    var q_reg = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    var do_reg = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    for e in range(ept):
        q_reg[e] = Float32(q[q_base + my_start + e])
        do_reg[e] = grad_output[q_base + my_start + e]

    var dq_acc = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    for e in range(ept):
        dq_acc[e] = 0.0

    var kp_start_dq = 0
    if window_size > 0 and qp > window_size:
        kp_start_dq = qp - window_size
    for kp in range(kp_start_dq, qp + 1):
        var k_base = (b * T * num_heads + kp * num_heads + h) * head_dim
        var v_base = k_base

        # Recompute Q·K via warp reduction
        var partial_qk: Float32 = 0.0
        for e in range(ept):
            partial_qk += q_reg[e] * Float32(k[k_base + my_start + e])
        var score = warp.sum(partial_qk) * scale
        var a_val = exp(score - lse_val)

        # dA = dO·V via warp reduction
        var partial_dov: Float32 = 0.0
        for e in range(ept):
            partial_dov += do_reg[e] * Float32(v[v_base + my_start + e])
        var da = warp.sum(partial_dov)

        # dS = A * (dA - D)
        var ds = a_val * (da - d_val)

        # Accumulate dQ (each thread handles its ept elements)
        for e in range(ept):
            dq_acc[e] += ds * Float32(k[k_base + my_start + e])

    # Store
    for e in range(ept):
        grad_q[q_base + my_start + e] = dq_acc[e] * scale


def flash_attn_bwd_dkv(
    grad_k: UnsafePointer[Float32, MutAnyOrigin],
    grad_v: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[BFloat16, MutAnyOrigin],
    k: UnsafePointer[BFloat16, MutAnyOrigin],
    v: UnsafePointer[BFloat16, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    d_buf: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    T: Int,
    num_heads: Int,
    head_dim: Int,
    window_size: Int,
):
    """Warp-parallel dK/dV: one warp per key position, recomputing A on the fly."""
    var ws = Int(WARP_SIZE)
    var lane = Int(thread_idx.x) % ws
    var warp_id = Int(thread_idx.x) // ws
    var warps_in_block = Int(block_dim.x) // ws
    var key_idx = Int(block_idx.x) * warps_in_block + warp_id
    var total = B * T * num_heads
    if key_idx >= total:
        return

    var bt = key_idx // num_heads
    var h = key_idx % num_heads
    var b = bt // T
    var kp = bt % T

    var k_base = key_idx * head_dim
    var scale = 1.0 / sqrt(Float32(head_dim))
    var ept = head_dim // ws
    var my_start = lane * ept

    # Load K and V into registers (bf16 → f32)
    var k_reg = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    var v_reg = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    for e in range(ept):
        k_reg[e] = Float32(k[k_base + my_start + e])
        v_reg[e] = Float32(v[k_base + my_start + e])

    var dk_acc = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    var dv_acc = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    for e in range(ept):
        dk_acc[e] = 0.0
        dv_acc[e] = 0.0

    var qp_start = kp
    var qp_end = T
    if window_size > 0 and kp + window_size < T:
        qp_end = kp + window_size + 1

    for qp in range(qp_start, qp_end):
        var q_idx = b * T * num_heads + qp * num_heads + h
        var q_base = q_idx * head_dim
        var lse_val = lse[q_idx]
        var d_val = d_buf[q_idx]

        # Q·K via warp reduction
        var partial_qk: Float32 = 0.0
        for e in range(ept):
            partial_qk += k_reg[e] * Float32(q[q_base + my_start + e])
        var score = warp.sum(partial_qk) * scale
        var a_val = exp(score - lse_val)

        # dV accumulation (each thread handles its ept elements)
        for e in range(ept):
            dv_acc[e] += a_val * grad_output[q_base + my_start + e]

        # dO·V via warp reduction for dA
        var partial_dov: Float32 = 0.0
        for e in range(ept):
            partial_dov += v_reg[e] * grad_output[q_base + my_start + e]
        var da = warp.sum(partial_dov)

        var ds = a_val * (da - d_val)

        # dK accumulation
        for e in range(ept):
            dk_acc[e] += ds * Float32(q[q_base + my_start + e])

    # Store
    for e in range(ept):
        grad_k[k_base + my_start + e] = dk_acc[e] * scale
        grad_v[k_base + my_start + e] = dv_acc[e]


# ── Elementwise ──────────────────────────────────────────────────────────────


def relu_squared_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
    size: Int,
):
    """relu(x)^2."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var x = Float32(input[tid])
    var r = x if x > 0.0 else Float32(0.0)
    output[tid] = BFloat16(r * r)


def softcap_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
    cap: Float32,
    size: Int,
):
    """cap * tanh(x / cap)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = BFloat16(cap * tanh(Float32(input[tid]) / cap))


# ── Value Embeddings ─────────────────────────────────────────────────────────


def ve_gate_fwd(
    gate_out: UnsafePointer[BFloat16, MutAnyOrigin],
    x_norm: UnsafePointer[BFloat16, MutAnyOrigin],
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
        acc += gate_weight[h * gate_channels + c] * Float32(x_norm[t * embd_dim + c])
    # 2 * sigmoid(x) = 2 / (1 + exp(-x))
    var sig = 1.0 / (1.0 + exp(-acc))
    gate_out[tid] = BFloat16(2.0 * sig)


def ve_apply_fwd(
    v: UnsafePointer[BFloat16, MutAnyOrigin],
    gate: UnsafePointer[BFloat16, MutAnyOrigin],
    ve: UnsafePointer[BFloat16, MutAnyOrigin],
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
    var g = Float32(gate[th])
    v[tid] = BFloat16(Float32(v[tid]) + g * Float32(ve[tid]))


# ── Residual Connections ─────────────────────────────────────────────────────


def add_residual_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    residual: UnsafePointer[BFloat16, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
    size: Int,
):
    """output = residual + input."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = BFloat16(Float32(residual[tid]) + Float32(input[tid]))


def add_residual_fwd_f32(
    output: UnsafePointer[Float32, MutAnyOrigin],
    residual: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """output = residual + input (float32 variant for gradient accumulation)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = residual[tid] + input[tid]


def scaled_add_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    a: UnsafePointer[BFloat16, MutAnyOrigin],
    b: UnsafePointer[BFloat16, MutAnyOrigin],
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
    output[tid] = BFloat16(sa * Float32(a[tid]) + sb * Float32(b[tid]))


# ── Softmax ──────────────────────────────────────────────────────────────────


def softmax_fwd(
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
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
        var val = Float32(input[row_offset + i])
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
        var val = exp(Float32(input[row_offset + i]) - row_max)
        output[row_offset + i] = BFloat16(val)
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
        output[row_offset + i] = BFloat16(Float32(output[row_offset + i]) / row_sum)


# ── Cross-Entropy ────────────────────────────────────────────────────────────


def cross_entropy_fwd(
    loss_out: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[BFloat16, MutAnyOrigin],
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
        var val = Float32(logits[row_offset + i])
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
        s += exp(Float32(logits[row_offset + i]) - row_max)
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
            loss_out[token] = -Float32(logits[row_offset + target]) + log_sum_exp
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
    logits: UnsafePointer[BFloat16, MutAnyOrigin],
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
        var val = Float32(logits[row_offset + i])
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
        s += exp(Float32(logits[row_offset + i]) - row_max)
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
        var softmax_val = exp(Float32(logits[row_offset + i]) - row_max) / row_sum
        if i == target:
            grad_logits[row_offset + i] = (softmax_val - 1.0) * loss_scale
        else:
            grad_logits[row_offset + i] = softmax_val * loss_scale


# ── Backward: Linear ────────────────────────────────────────────────────────


def linear_bwd_dx(
    grad_input: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[BFloat16, MutAnyOrigin],
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

        # Load W[n_chunk, k_range] into b_smem[TILE_K, TILE_N] (bf16 → f32)
        for li in range(tid, TILE_K * TILE_N, MM_THREADS):
            var ln = li // TILE_N
            var lk = li % TILE_N
            var gn = n_base + ln
            var gk = block_k * TILE_N + lk
            b_smem[li] = Float32(weight[gn * K + gk]) if gn < N and gk < K else Float32(0)
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
    input: UnsafePointer[BFloat16, MutAnyOrigin],
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

        # Load X[m_chunk, k_range] (bf16 → f32)
        for li in range(tid, TILE_K * TILE_N, MM_THREADS):
            var lm = li // TILE_N
            var lk = li % TILE_N
            var gm = m_base + lm
            var gk = block_k * TILE_N + lk
            b_smem[li] = Float32(input[gm * K + gk]) if gm < M and gk < K else Float32(0)
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
    input: UnsafePointer[BFloat16, MutAnyOrigin],
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
        dot += grad_output[row_offset + i] * Float32(input[row_offset + i])
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
        var x = Float32(input[row_offset + i])
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
    input: UnsafePointer[BFloat16, MutAnyOrigin],
    size: Int,
):
    """d/dx relu(x)^2 = 2*x*dy if x > 0, else 0."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var x = Float32(input[tid])
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
    gate: UnsafePointer[BFloat16, MutAnyOrigin],
    ve: UnsafePointer[BFloat16, MutAnyOrigin],
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
    var g = Float32(gate[row])

    var dg: Float32 = 0.0
    for d in range(Int(tid), head_dim, Int(THREADS_PER_BLOCK)):
        var gv = grad_v[base + d]
        grad_ve[base + d] = g * gv
        dg += Float32(ve[base + d]) * gv
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
    gate: UnsafePointer[BFloat16, MutAnyOrigin],
    x_norm: UnsafePointer[BFloat16, MutAnyOrigin],
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

    var g = Float32(gate[tid])
    var dz = grad_gate[tid] * g * (1.0 - g / 2.0)

    comptime ord = Consistency.RELEASE if is_apple_gpu() else Consistency.SEQUENTIAL
    for c in range(gate_channels):
        var x_val = Float32(x_norm[t * embd_dim + c])
        _ = Atomic.fetch_add[ordering=ord](
            grad_gate_weight + h * gate_channels + c, dz * x_val)
    for c in range(gate_channels):
        _ = Atomic.fetch_add[ordering=ord](
            grad_x_norm_out + t * embd_dim + c, dz * gate_weight[h * gate_channels + c])


def scalar_grad_reduce(
    grad_scalar: UnsafePointer[Float32, MutAnyOrigin],
    grad_output: UnsafePointer[Float32, MutAnyOrigin],
    input: UnsafePointer[BFloat16, MutAnyOrigin],
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
        s += grad_output[i] * Float32(input[i])
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
