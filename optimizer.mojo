# AdamW and Muon optimizer GPU kernels.
#
# AdamW: standard bias-corrected Adam with decoupled weight decay.
# Muon: Nesterov momentum → Polar Express orthogonalization → cautious update.

from std.math import ceildiv, sqrt
from std.gpu import barrier, block_dim, block_idx, thread_idx
from std.gpu.primitives import warp
from std.gpu.globals import WARP_SIZE
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.os.atomic import Atomic, Consistency
from std.sys.info import is_apple_gpu

from kernels.ops import (
    linear_fwd_f32, linear_bwd_dx_f32, linear_bwd_dw_f32,
    TILE_M, TILE_N, TILE_K, MM_THREADS,
)
from config import POLAR_COEFF_A, POLAR_COEFF_B, POLAR_COEFF_C

comptime BLOCK: Int = 256


# ── AdamW GPU kernel ─────────────────────────────────────────────────────────

def adamw_kernel(
    params: UnsafePointer[BFloat16, MutAnyOrigin],
    grads: UnsafePointer[Float32, MutAnyOrigin],
    exp_avg: UnsafePointer[Float32, MutAnyOrigin],
    exp_avg_sq: UnsafePointer[Float32, MutAnyOrigin],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bias_corr1: Float32,
    bias_corr2: Float32,
    size: Int,
):
    """AdamW step. Params bf16, grads/moments f32. Bias correction pre-computed on CPU."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return

    var p = Float32(params[tid])
    var g = grads[tid]

    p = p * (1.0 - lr * weight_decay)

    var m = beta1 * exp_avg[tid] + (1.0 - beta1) * g
    exp_avg[tid] = m

    var v = beta2 * exp_avg_sq[tid] + (1.0 - beta2) * g * g
    exp_avg_sq[tid] = v

    params[tid] = BFloat16(p - lr * (m / bias_corr1) / (sqrt(v / bias_corr2) + eps))


def adamw_kernel_f32(
    params: UnsafePointer[Float32, MutAnyOrigin],
    grads: UnsafePointer[Float32, MutAnyOrigin],
    exp_avg: UnsafePointer[Float32, MutAnyOrigin],
    exp_avg_sq: UnsafePointer[Float32, MutAnyOrigin],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bias_corr1: Float32,
    bias_corr2: Float32,
    size: Int,
):
    """AdamW step for float32 params (scalar parameters like resid_lambdas)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var p = params[tid]
    var g = grads[tid]
    p = p * (1.0 - lr * weight_decay)
    var m = beta1 * exp_avg[tid] + (1.0 - beta1) * g
    exp_avg[tid] = m
    var v = beta2 * exp_avg_sq[tid] + (1.0 - beta2) * g * g
    exp_avg_sq[tid] = v
    params[tid] = p - lr * (m / bias_corr1) / (sqrt(v / bias_corr2) + eps)


# ── AdamW state ──────────────────────────────────────────────────────────────

struct AdamWState:
    """Per-group first/second moment buffers and hyperparameters."""

    var exp_avg: List[DeviceBuffer[DType.float32]]
    var exp_avg_sq: List[DeviceBuffer[DType.float32]]
    var sizes: List[Int]
    var lrs: List[Float64]
    var betas1: List[Float64]
    var betas2: List[Float64]
    var eps: List[Float64]
    var weight_decays: List[Float64]
    var step: Int

    fn __init__(out self):
        self.exp_avg = List[DeviceBuffer[DType.float32]]()
        self.exp_avg_sq = List[DeviceBuffer[DType.float32]]()
        self.sizes = List[Int]()
        self.lrs = List[Float64]()
        self.betas1 = List[Float64]()
        self.betas2 = List[Float64]()
        self.eps = List[Float64]()
        self.weight_decays = List[Float64]()
        self.step = 0

    fn add_param_group(
        mut self, ctx: DeviceContext, size: Int,
        lr: Float64, beta1: Float64, beta2: Float64, eps: Float64, weight_decay: Float64,
    ) raises:
        var m = ctx.enqueue_create_buffer[DType.float32](size)
        var v = ctx.enqueue_create_buffer[DType.float32](size)
        m.enqueue_fill(0)
        v.enqueue_fill(0)
        self.exp_avg.append(m)
        self.exp_avg_sq.append(v)
        self.sizes.append(size)
        self.lrs.append(lr)
        self.betas1.append(beta1)
        self.betas2.append(beta2)
        self.eps.append(eps)
        self.weight_decays.append(weight_decay)

    fn step_group_f32(
        self, ctx: DeviceContext, group_idx: Int,
        params: DeviceBuffer[DType.float32], grads: DeviceBuffer[DType.float32],
        lr_multiplier: Float64,
    ) raises:
        """AdamW step for float32 params (scalar parameters)."""
        var size = self.sizes[group_idx]
        var lr = Float32(self.lrs[group_idx] * lr_multiplier)
        var b1 = Float32(self.betas1[group_idx])
        var b2 = Float32(self.betas2[group_idx])
        var bc1: Float32 = 1.0
        var bc2: Float32 = 1.0
        for _ in range(self.step):
            bc1 *= b1
            bc2 *= b2
        ctx.enqueue_function[adamw_kernel_f32, adamw_kernel_f32](
            params, grads,
            self.exp_avg[group_idx], self.exp_avg_sq[group_idx],
            lr, b1, b2, Float32(self.eps[group_idx]),
            Float32(self.weight_decays[group_idx]),
            1.0 - bc1, 1.0 - bc2, size,
            grid_dim=ceildiv(size, BLOCK), block_dim=BLOCK,
        )

    fn step_group(
        self, ctx: DeviceContext, group_idx: Int,
        params: DeviceBuffer[DType.bfloat16], grads: DeviceBuffer[DType.float32],
        lr_multiplier: Float64,
    ) raises:
        var size = self.sizes[group_idx]
        var lr = Float32(self.lrs[group_idx] * lr_multiplier)
        var b1 = Float32(self.betas1[group_idx])
        var b2 = Float32(self.betas2[group_idx])

        # Bias correction: beta^step computed on CPU
        var bc1: Float32 = 1.0
        var bc2: Float32 = 1.0
        for _ in range(self.step):
            bc1 *= b1
            bc2 *= b2

        ctx.enqueue_function[adamw_kernel, adamw_kernel](
            params, grads,
            self.exp_avg[group_idx], self.exp_avg_sq[group_idx],
            lr, b1, b2, Float32(self.eps[group_idx]),
            Float32(self.weight_decays[group_idx]),
            1.0 - bc1, 1.0 - bc2, size,
            grid_dim=ceildiv(size, BLOCK), block_dim=BLOCK,
        )


# ── Muon GPU kernels ─────────────────────────────────────────────────────────

def nesterov_momentum_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    grads: UnsafePointer[Float32, MutAnyOrigin],
    momentum_buf: UnsafePointer[Float32, MutAnyOrigin],
    momentum: Float32,
    size: Int,
):
    """Nesterov momentum: buf.lerp_(grad, 1-mu); out = grad.lerp_(buf, mu)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var g = grads[tid]
    var m = momentum * momentum_buf[tid] + (1.0 - momentum) * g
    momentum_buf[tid] = m
    output[tid] = (1.0 - momentum) * g + momentum * m


def normalize_frobenius_kernel(
    data: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """In-place: data /= (||data||_F * 1.02 + 1e-6). Fused norm + scale, single block."""
    var tid = thread_idx.x
    comptime tpb: UInt = 256
    var smem = stack_allocation[Int(tpb), Float32, address_space=AddressSpace.SHARED]()

    var s: Float32 = 0.0
    for i in range(Int(tid), size, Int(tpb)):
        var v = data[i]
        s += v * v
    smem[tid] = s
    barrier()

    var active = tpb
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < UInt(active):
            smem[tid] += smem[UInt(tid) + active]
        barrier()
    if tid < UInt(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            smem[0] = wv
    barrier()

    var inv_norm = 1.0 / (sqrt(smem[0][0]) * 1.02 + 1e-6)
    for i in range(Int(tid), size, Int(tpb)):
        data[i] = data[i] * inv_norm


def scale_add_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    sa: Float32, sb: Float32,
    size: Int,
):
    """output = sa*a + sb*b."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = sa * a[tid] + sb * b[tid]


def normuon_variance_reduction(
    g: UnsafePointer[Float32, MutAnyOrigin],
    second_momentum: UnsafePointer[Float32, MutAnyOrigin],
    beta2: Float32,
    nrows: Int, ncols: Int,
    is_tall: Int,
):
    """NorMuon per-row (tall) or per-col (wide) variance reduction. One block.

    For tall matrices (rows >= cols): reduces over cols, scales per row.
    For wide matrices (rows < cols): reduces over rows, scales per col.
    """
    var tid = Int(thread_idx.x)
    comptime tpb: UInt = 256
    var smem = stack_allocation[Int(tpb), Float32, address_space=AddressSpace.SHARED]()

    var n_keep = nrows if is_tall == 1 else ncols
    var n_reduce = ncols if is_tall == 1 else nrows

    # Each thread handles up to 4 elements
    var v_mean_local = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    var sq_sum_local = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    var step_local = stack_allocation[4, Float32, address_space=AddressSpace.LOCAL]()
    var num_my = 0

    # Phase 1: compute per-element variance and accumulate v_norm_sq
    var my_v_norm_sq: Float32 = 0.0
    for k in range(tid, n_keep, Int(tpb)):
        var sq_sum: Float32 = 0.0
        for j in range(n_reduce):
            var idx: Int
            if is_tall == 1:
                idx = k * ncols + j
            else:
                idx = j * ncols + k
            var val = g[idx]
            sq_sum += val * val
        var v_mean = sq_sum / Float32(n_reduce)
        v_mean_local[num_my] = v_mean
        sq_sum_local[num_my] = sq_sum
        my_v_norm_sq += sq_sum
        num_my += 1

    # Reduce v_norm_sq across threads
    smem[tid] = my_v_norm_sq
    barrier()
    var active = tpb
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < Int(active):
            smem[tid] += smem[UInt(tid) + active]
        barrier()
    if tid < Int(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var v_norm = sqrt(smem[0][0])

    # Phase 2: update second momentum, compute step_size, accumulate v_norm_new_sq
    var my_v_norm_new_sq: Float32 = 0.0
    for i in range(num_my):
        var k = tid + i * Int(tpb)
        var sm = second_momentum[k]
        sm = beta2 * sm + (1.0 - beta2) * v_mean_local[i]
        second_momentum[k] = sm
        var clamped = sm if sm > 1e-10 else Float32(1e-10)
        var ss = 1.0 / sqrt(clamped)
        step_local[i] = ss
        my_v_norm_new_sq += sq_sum_local[i] * ss * ss

    # Reduce v_norm_new_sq
    smem[tid] = my_v_norm_new_sq
    barrier()
    active = tpb
    while active > UInt(WARP_SIZE):
        active >>= 1
        if tid < Int(active):
            smem[tid] += smem[UInt(tid) + active]
        barrier()
    if tid < Int(WARP_SIZE):
        var wv: Float32 = smem[tid][0]
        wv = warp.sum(wv)
        if tid == 0:
            smem[0] = wv
    barrier()
    var v_norm_new = sqrt(smem[0][0])
    var norm_ratio = v_norm / (v_norm_new if v_norm_new > 1e-10 else Float32(1e-10))

    # Phase 3: apply per-element scale
    for i in range(num_my):
        var k = tid + i * Int(tpb)
        var final_scale = step_local[i] * norm_ratio
        for j in range(n_reduce):
            var idx: Int
            if is_tall == 1:
                idx = k * ncols + j
            else:
                idx = j * ncols + k
            g[idx] = g[idx] * final_scale


def muon_update_kernel(
    params: UnsafePointer[BFloat16, MutAnyOrigin],
    grads: UnsafePointer[Float32, MutAnyOrigin],
    lr: Float32, wd: Float32,
    size: Int,
):
    """Cautious update: params bf16, grads f32. Decay only when gradient and param have same sign."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var p = Float32(params[tid])
    var g = grads[tid]
    var decay = wd * p if (g * p) >= 0.0 else Float32(0.0)
    params[tid] = BFloat16(p - lr * (g + decay))


# ── Muon state ───────────────────────────────────────────────────────────────

struct MuonState:
    """Muon optimizer: Nesterov momentum → Polar Express → cautious update.

    Each registered parameter is a [rows, cols] matrix. The Polar Express
    orthogonalization computes 5 iterations of a Cayley-transform approximation
    to the polar decomposition, using small (min_dim × min_dim) matmuls.
    """

    var momentum_bufs: List[DeviceBuffer[DType.float32]]
    var second_momentum_bufs: List[DeviceBuffer[DType.float32]]  # NorMuon
    var temp_g: List[DeviceBuffer[DType.float32]]
    var temp_a: List[DeviceBuffer[DType.float32]]   # X^T@X or X@X^T
    var temp_b: List[DeviceBuffer[DType.float32]]   # b*A + c*(A@A)
    var temp_aa: List[DeviceBuffer[DType.float32]]   # A@A
    var rows: List[Int]
    var cols: List[Int]
    var initial_lr: Float64

    fn __init__(out self, ctx: DeviceContext, initial_lr: Float64) raises:
        self.momentum_bufs = List[DeviceBuffer[DType.float32]]()
        self.second_momentum_bufs = List[DeviceBuffer[DType.float32]]()
        self.temp_g = List[DeviceBuffer[DType.float32]]()
        self.temp_a = List[DeviceBuffer[DType.float32]]()
        self.temp_b = List[DeviceBuffer[DType.float32]]()
        self.temp_aa = List[DeviceBuffer[DType.float32]]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.initial_lr = initial_lr

    fn add_param(mut self, ctx: DeviceContext, nrows: Int, ncols: Int) raises:
        var size = nrows * ncols
        var mbuf = ctx.enqueue_create_buffer[DType.float32](size)
        mbuf.enqueue_fill(0)
        self.momentum_bufs.append(mbuf)
        # NorMuon: per-row (tall) or per-col (wide) second momentum
        var n_keep = nrows if nrows >= ncols else ncols
        var sm2 = ctx.enqueue_create_buffer[DType.float32](n_keep)
        sm2.enqueue_fill(0)
        self.second_momentum_bufs.append(sm2)
        self.temp_g.append(ctx.enqueue_create_buffer[DType.float32](size))
        var small = ncols if nrows >= ncols else nrows
        self.temp_a.append(ctx.enqueue_create_buffer[DType.float32](small * small))
        self.temp_b.append(ctx.enqueue_create_buffer[DType.float32](small * small))
        self.temp_aa.append(ctx.enqueue_create_buffer[DType.float32](small * small))
        self.rows.append(nrows)
        self.cols.append(ncols)

    fn step(
        self, ctx: DeviceContext, idx: Int,
        params: DeviceBuffer[DType.bfloat16], grads: DeviceBuffer[DType.float32],
        lr_multiplier: Float64, momentum: Float64, weight_decay: Float64,
    ) raises:
        var nrows = self.rows[idx]
        var ncols = self.cols[idx]
        var size = nrows * ncols
        var small = ncols if nrows >= ncols else nrows
        var ss = small * small

        # Scale LR by sqrt(aspect_ratio) for non-square matrices
        var aspect = Float64(nrows) / Float64(ncols)
        var lr = Float32(self.initial_lr * lr_multiplier * (sqrt(aspect) if aspect > 1.0 else 1.0))

        # 1. Nesterov momentum
        ctx.enqueue_function[nesterov_momentum_kernel, nesterov_momentum_kernel](
            self.temp_g[idx], grads, self.momentum_bufs[idx],
            Float32(momentum), size,
            grid_dim=ceildiv(size, BLOCK), block_dim=BLOCK,
        )

        # 2. Polar Express: normalize then 5 iterations of Cayley transform
        ctx.enqueue_function[normalize_frobenius_kernel, normalize_frobenius_kernel](
            self.temp_g[idx], size, grid_dim=1, block_dim=BLOCK,
        )

        var tall = nrows >= ncols
        for pe_iter in range(5):
            var ca = Float32(POLAR_COEFF_A(pe_iter))
            var cb = Float32(POLAR_COEFF_B(pe_iter))
            var cc = Float32(POLAR_COEFF_C(pe_iter))

            # A = X^T@X (tall) or X@X^T (wide)
            if tall:
                self.temp_a[idx].enqueue_fill(0)
                ctx.enqueue_function[linear_bwd_dw_f32, linear_bwd_dw_f32](
                    self.temp_a[idx], self.temp_g[idx], self.temp_g[idx],
                    nrows, ncols, ncols,
                    grid_dim=(ceildiv(ncols, TILE_N), ceildiv(ncols, TILE_M)),
                    block_dim=MM_THREADS,
                )
            else:
                ctx.enqueue_function[linear_fwd_f32, linear_fwd_f32](
                    self.temp_a[idx], self.temp_g[idx], self.temp_g[idx],
                    nrows, nrows, ncols,
                    grid_dim=(ceildiv(nrows, TILE_N), ceildiv(nrows, TILE_M)),
                    block_dim=MM_THREADS,
                )

            # B = b*A + c*(A@A)
            ctx.enqueue_function[linear_bwd_dx_f32, linear_bwd_dx_f32](
                self.temp_aa[idx], self.temp_a[idx], self.temp_a[idx],
                small, small, small,
                grid_dim=(ceildiv(small, TILE_N), ceildiv(small, TILE_M)),
                block_dim=MM_THREADS,
            )
            ctx.enqueue_function[scale_add_kernel, scale_add_kernel](
                self.temp_b[idx], self.temp_a[idx], self.temp_aa[idx], cb, cc, ss,
                grid_dim=ceildiv(ss, BLOCK), block_dim=BLOCK,
            )

            # X_new = a*X + X@B (tall) or a*X + B@X (wide)
            # Uses `grads` buffer as scratch (safe: we're done with it)
            if tall:
                ctx.enqueue_function[linear_bwd_dx_f32, linear_bwd_dx_f32](
                    grads, self.temp_g[idx], self.temp_b[idx],
                    nrows, ncols, ncols,
                    grid_dim=(ceildiv(ncols, TILE_N), ceildiv(nrows, TILE_M)),
                    block_dim=MM_THREADS,
                )
            else:
                ctx.enqueue_function[linear_bwd_dx_f32, linear_bwd_dx_f32](
                    grads, self.temp_b[idx], self.temp_g[idx],
                    nrows, nrows, ncols,
                    grid_dim=(ceildiv(ncols, TILE_N), ceildiv(nrows, TILE_M)),
                    block_dim=MM_THREADS,
                )
            ctx.enqueue_function[scale_add_kernel, scale_add_kernel](
                self.temp_g[idx], self.temp_g[idx], grads, ca, Float32(1.0), size,
                grid_dim=ceildiv(size, BLOCK), block_dim=BLOCK,
            )

        # 3. NorMuon variance reduction
        ctx.enqueue_function[normuon_variance_reduction, normuon_variance_reduction](
            self.temp_g[idx], self.second_momentum_bufs[idx],
            Float32(0.95),  # beta2
            nrows, ncols,
            1 if tall else 0,
            grid_dim=1, block_dim=BLOCK,
        )

        # 4. Cautious update with weight decay
        ctx.enqueue_function[muon_update_kernel, muon_update_kernel](
            params, self.temp_g[idx], lr, Float32(weight_decay), size,
            grid_dim=ceildiv(size, BLOCK), block_dim=BLOCK,
        )
