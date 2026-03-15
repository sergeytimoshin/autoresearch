# GPT model: weights, forward pass, backward pass.

# ── Imports ──────────────────────────────────────────────────────────────

from std.sys import has_accelerator
from std.math import ceildiv, sqrt, cos, sin
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx

from config import (
    GPTConfig,
    MAX_SEQ_LEN,
    MLP_EXPANSION,
    VE_GATE_CHANNELS,
    SOFTCAP,
    ROPE_BASE,
)
from kernels.ops import (
    embedding_fwd,
    rmsnorm_fwd,
    linear_fwd, TILE_M, TILE_N, TILE_K, MM_THREADS,
    rope_fwd,
    flash_attn_fwd,
    flash_attn_bwd_precompute_d,
    flash_attn_bwd_dq,
    flash_attn_bwd_dkv,
    relu_squared_fwd,
    softcap_fwd,
    ve_gate_fwd,
    ve_apply_fwd,
    ve_apply_bwd,
    ve_gate_bwd,
    add_residual_fwd,
    scaled_add_fwd,
    cross_entropy_fwd,
    mean_reduce,
    # Backward kernels
    cross_entropy_softcap_bwd,
    linear_bwd_dx,
    linear_bwd_dw,
    rmsnorm_bwd,
    rope_bwd,
    relu_squared_bwd,
    embedding_bwd,
    scaled_add_bwd,
    scalar_grad_reduce,
    zero_buffer,
)

# ── Constants ────────────────────────────────────────────────────────────

comptime LAUNCH_BLOCK = 256

# ── RoPE Precomputation ─────────────────────────────────────────────────


fn precompute_rope(
    ctx: DeviceContext,
    seq_len: Int,
    head_dim: Int,
    cos_buf: DeviceBuffer[DType.float32],
    sin_buf: DeviceBuffer[DType.float32],
) raises:
    """Precompute cos/sin for RoPE. Fills [seq_len, head_dim//2] buffers."""
    var half_dim = head_dim // 2
    var total = seq_len * half_dim

    var cos_host = ctx.enqueue_create_host_buffer[DType.float32](total)
    var sin_host = ctx.enqueue_create_host_buffer[DType.float32](total)
    ctx.synchronize()

    # inv_freq[d] = 1 / (base ^ (2d / head_dim))
    for t in range(seq_len):
        for d in range(half_dim):
            var freq = Float64(t) / (ROPE_BASE ** (Float64(2 * d) / Float64(head_dim)))
            var idx = t * half_dim + d
            cos_host[idx] = Float32(cos(freq))
            sin_host[idx] = Float32(sin(freq))

    ctx.enqueue_copy(dst_buf=cos_buf, src_buf=cos_host)
    ctx.enqueue_copy(dst_buf=sin_buf, src_buf=sin_host)

# ── Weight Initialization Kernels ────────────────────────────────────────


def fill_normal_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    seed_base: UInt64,
    std: Float32,
    size: Int,
):
    """Fill with pseudo-random normal(0, std) using simple xorshift+Box-Muller.

    This is a crude PRNG for initialization -- not cryptographic quality.
    """
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    # Simple xorshift64 PRNG
    var state = seed_base + UInt64(tid) * 6364136223846793005 + 1442695040888963407
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    var u1 = Float32(state & 0xFFFFFF) / Float32(0x1000000) + 1e-10
    state = state * 6364136223846793005 + 1442695040888963407
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    var u2 = Float32(state & 0xFFFFFF) / Float32(0x1000000)

    # Box-Muller transform
    from std.math import sqrt, log, cos
    var PI: Float32 = 3.14159265358979323846
    var z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)
    output[tid] = z * std


def fill_uniform_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    seed_base: UInt64,
    low: Float32,
    high: Float32,
    size: Int,
):
    """Fill with uniform(low, high) using xorshift."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    var state = seed_base + UInt64(tid) * 6364136223846793005 + 1442695040888963407
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    var u = Float32(state & 0xFFFFFF) / Float32(0x1000000)
    output[tid] = low + u * (high - low)


def fill_zeros_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    """Fill buffer with zeros."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = 0.0


def fill_value_kernel(
    output: UnsafePointer[Float32, MutAnyOrigin],
    value: Float32,
    size: Int,
):
    """Fill buffer with a constant value."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    output[tid] = value

# ── Launch Helpers ───────────────────────────────────────────────────────


fn launch_linear(
    ctx: DeviceContext,
    output: DeviceBuffer[DType.float32],
    input: DeviceBuffer[DType.float32],
    weight: DeviceBuffer[DType.float32],
    M: Int, N: Int, K: Int,
) raises:
    """Launch tiled matmul: Y[m,n] = sum_k X[m,k] * W[n,k]."""
    ctx.enqueue_function[linear_fwd, linear_fwd](
        output, input, weight, M, N, K,
        grid_dim=(ceildiv(N, TILE_N), ceildiv(M, TILE_M)),
        block_dim=MM_THREADS,
    )


fn launch_linear_bwd_dx(
    ctx: DeviceContext,
    grad_input: DeviceBuffer[DType.float32],
    grad_output: DeviceBuffer[DType.float32],
    weight: DeviceBuffer[DType.float32],
    M: Int, N: Int, K: Int,
) raises:
    """Launch tiled NN matmul: dX[m,k] = sum_n dY[m,n] * W[n,k]."""
    ctx.enqueue_function[linear_bwd_dx, linear_bwd_dx](
        grad_input, grad_output, weight, M, N, K,
        grid_dim=(ceildiv(K, TILE_N), ceildiv(M, TILE_M)),
        block_dim=MM_THREADS,
    )


fn launch_linear_bwd_dw(
    ctx: DeviceContext,
    grad_weight: DeviceBuffer[DType.float32],
    grad_output: DeviceBuffer[DType.float32],
    input: DeviceBuffer[DType.float32],
    M: Int, N: Int, K: Int,
) raises:
    """Launch tiled TN matmul: dW[n,k] += sum_m dY[m,n] * X[m,k]."""
    ctx.enqueue_function[linear_bwd_dw, linear_bwd_dw](
        grad_weight, grad_output, input, M, N, K,
        grid_dim=(ceildiv(K, TILE_N), ceildiv(N, TILE_M)),
        block_dim=MM_THREADS,
    )

# ── ModelWeights ─────────────────────────────────────────────────────────


struct ModelWeights:
    """All model parameters stored as flat GPU buffers (float32).

    Layout matches train.py's GPT model:
    - wte: [vocab_size, n_embd] -- token embeddings
    - lm_head: [vocab_size, n_embd] -- output projection
    - value_embeds: n_layer/2 x [vocab_size, kv_dim] -- value embeddings (alternating layers)
    - resid_lambdas: [n_layer] -- residual scaling factors
    - x0_lambdas: [n_layer] -- skip connection scaling factors
    - Per block: c_q, c_k, c_v: [n_embd, n_embd], c_proj: [n_embd, n_embd]
    - Per block: c_fc: [4*n_embd, n_embd], mlp_proj: [n_embd, 4*n_embd]
    - Per block (if has_ve): ve_gate: [n_kv_head, VE_GATE_CHANNELS]
    """

    var config: GPTConfig

    # Embeddings
    var wte: DeviceBuffer[DType.float32]
    var lm_head: DeviceBuffer[DType.float32]

    # Per-layer scalars
    var resid_lambdas: DeviceBuffer[DType.float32]
    var x0_lambdas: DeviceBuffer[DType.float32]

    # Per-block linear weights (stored as List of buffers)
    var c_q: List[DeviceBuffer[DType.float32]]
    var c_k: List[DeviceBuffer[DType.float32]]
    var c_v: List[DeviceBuffer[DType.float32]]
    var c_proj: List[DeviceBuffer[DType.float32]]
    var c_fc: List[DeviceBuffer[DType.float32]]
    var mlp_proj: List[DeviceBuffer[DType.float32]]

    # Value embeddings and gates (only for layers with has_ve)
    var value_embeds: List[DeviceBuffer[DType.float32]]
    var ve_gate: List[DeviceBuffer[DType.float32]]

    # RoPE buffers
    var rope_cos: DeviceBuffer[DType.float32]
    var rope_sin: DeviceBuffer[DType.float32]

    # Gradient buffers (same shapes as weights)
    var grad_wte: DeviceBuffer[DType.float32]
    var grad_lm_head: DeviceBuffer[DType.float32]
    var grad_resid_lambdas: DeviceBuffer[DType.float32]
    var grad_x0_lambdas: DeviceBuffer[DType.float32]
    var grad_c_q: List[DeviceBuffer[DType.float32]]
    var grad_c_k: List[DeviceBuffer[DType.float32]]
    var grad_c_v: List[DeviceBuffer[DType.float32]]
    var grad_c_proj: List[DeviceBuffer[DType.float32]]
    var grad_c_fc: List[DeviceBuffer[DType.float32]]
    var grad_mlp_proj: List[DeviceBuffer[DType.float32]]
    var grad_value_embeds: List[DeviceBuffer[DType.float32]]
    var grad_ve_gate: List[DeviceBuffer[DType.float32]]

    fn __init__(out self, ctx: DeviceContext, config: GPTConfig) raises:
        """Allocate all weight buffers."""
        self.config = config
        var C = config.n_embd
        var V = config.vocab_size
        var L = config.n_layer
        var kv_dim = config.kv_dim
        var mlp_dim = MLP_EXPANSION * C
        var half_dim = config.head_dim // 2

        # Embeddings
        self.wte = ctx.enqueue_create_buffer[DType.float32](V * C)
        self.lm_head = ctx.enqueue_create_buffer[DType.float32](V * C)

        # Scalars
        self.resid_lambdas = ctx.enqueue_create_buffer[DType.float32](L)
        self.x0_lambdas = ctx.enqueue_create_buffer[DType.float32](L)

        # Per-block weights
        self.c_q = List[DeviceBuffer[DType.float32]]()
        self.c_k = List[DeviceBuffer[DType.float32]]()
        self.c_v = List[DeviceBuffer[DType.float32]]()
        self.c_proj = List[DeviceBuffer[DType.float32]]()
        self.c_fc = List[DeviceBuffer[DType.float32]]()
        self.mlp_proj = List[DeviceBuffer[DType.float32]]()
        self.value_embeds = List[DeviceBuffer[DType.float32]]()
        self.ve_gate = List[DeviceBuffer[DType.float32]]()

        for i in range(L):
            self.c_q.append(ctx.enqueue_create_buffer[DType.float32](C * C))
            self.c_k.append(ctx.enqueue_create_buffer[DType.float32](kv_dim * C))
            self.c_v.append(ctx.enqueue_create_buffer[DType.float32](kv_dim * C))
            self.c_proj.append(ctx.enqueue_create_buffer[DType.float32](C * C))
            self.c_fc.append(ctx.enqueue_create_buffer[DType.float32](mlp_dim * C))
            self.mlp_proj.append(ctx.enqueue_create_buffer[DType.float32](C * mlp_dim))
            if config.has_ve(i):
                self.value_embeds.append(ctx.enqueue_create_buffer[DType.float32](V * kv_dim))
                self.ve_gate.append(ctx.enqueue_create_buffer[DType.float32](config.n_kv_head * VE_GATE_CHANNELS))
            else:
                # Placeholder (1 element) for layers without VE
                self.value_embeds.append(ctx.enqueue_create_buffer[DType.float32](1))
                self.ve_gate.append(ctx.enqueue_create_buffer[DType.float32](1))

        # RoPE
        self.rope_cos = ctx.enqueue_create_buffer[DType.float32](MAX_SEQ_LEN * half_dim)
        self.rope_sin = ctx.enqueue_create_buffer[DType.float32](MAX_SEQ_LEN * half_dim)

        # Gradient buffers
        self.grad_wte = ctx.enqueue_create_buffer[DType.float32](V * C)
        self.grad_lm_head = ctx.enqueue_create_buffer[DType.float32](V * C)
        self.grad_resid_lambdas = ctx.enqueue_create_buffer[DType.float32](L)
        self.grad_x0_lambdas = ctx.enqueue_create_buffer[DType.float32](L)
        self.grad_c_q = List[DeviceBuffer[DType.float32]]()
        self.grad_c_k = List[DeviceBuffer[DType.float32]]()
        self.grad_c_v = List[DeviceBuffer[DType.float32]]()
        self.grad_c_proj = List[DeviceBuffer[DType.float32]]()
        self.grad_c_fc = List[DeviceBuffer[DType.float32]]()
        self.grad_mlp_proj = List[DeviceBuffer[DType.float32]]()
        self.grad_value_embeds = List[DeviceBuffer[DType.float32]]()
        self.grad_ve_gate = List[DeviceBuffer[DType.float32]]()
        for i in range(L):
            self.grad_c_q.append(ctx.enqueue_create_buffer[DType.float32](C * C))
            self.grad_c_k.append(ctx.enqueue_create_buffer[DType.float32](kv_dim * C))
            self.grad_c_v.append(ctx.enqueue_create_buffer[DType.float32](kv_dim * C))
            self.grad_c_proj.append(ctx.enqueue_create_buffer[DType.float32](C * C))
            self.grad_c_fc.append(ctx.enqueue_create_buffer[DType.float32](mlp_dim * C))
            self.grad_mlp_proj.append(ctx.enqueue_create_buffer[DType.float32](C * mlp_dim))
            if config.has_ve(i):
                self.grad_value_embeds.append(ctx.enqueue_create_buffer[DType.float32](V * kv_dim))
                self.grad_ve_gate.append(ctx.enqueue_create_buffer[DType.float32](config.n_kv_head * VE_GATE_CHANNELS))
            else:
                self.grad_value_embeds.append(ctx.enqueue_create_buffer[DType.float32](1))
                self.grad_ve_gate.append(ctx.enqueue_create_buffer[DType.float32](1))

    fn init_weights(self, ctx: DeviceContext) raises:
        """Initialize all weights matching train.py's init_weights()."""
        var C = self.config.n_embd
        var V = self.config.vocab_size
        var L = self.config.n_layer
        var kv_dim = self.config.kv_dim
        var mlp_dim = MLP_EXPANSION * C
        _ = self.config.head_dim // 2

        var s = Float32(sqrt(Float64(3)) * (Float64(C) ** -0.5))
        var seed: UInt64 = 42

        # wte: normal(0, 1)
        ctx.enqueue_function[fill_normal_kernel, fill_normal_kernel](
            self.wte, seed, Float32(1.0), V * C,
            grid_dim=ceildiv(V * C, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
        )
        seed += 1

        # lm_head: normal(0, 0.001)
        ctx.enqueue_function[fill_normal_kernel, fill_normal_kernel](
            self.lm_head, seed, Float32(0.001), V * C,
            grid_dim=ceildiv(V * C, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
        )
        seed += 1

        # resid_lambdas = 1.0, x0_lambdas = 0.1
        ctx.enqueue_function[fill_value_kernel, fill_value_kernel](
            self.resid_lambdas, Float32(1.0), L,
            grid_dim=1, block_dim=LAUNCH_BLOCK,
        )
        ctx.enqueue_function[fill_value_kernel, fill_value_kernel](
            self.x0_lambdas, Float32(0.1), L,
            grid_dim=1, block_dim=LAUNCH_BLOCK,
        )

        for i in range(L):
            # c_q, c_k, c_v: uniform(-s, s)
            ctx.enqueue_function[fill_uniform_kernel, fill_uniform_kernel](
                self.c_q[i], seed, -s, s, C * C,
                grid_dim=ceildiv(C * C, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
            )
            seed += 1
            ctx.enqueue_function[fill_uniform_kernel, fill_uniform_kernel](
                self.c_k[i], seed, -s, s, kv_dim * C,
                grid_dim=ceildiv(kv_dim * C, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
            )
            seed += 1
            ctx.enqueue_function[fill_uniform_kernel, fill_uniform_kernel](
                self.c_v[i], seed, -s, s, kv_dim * C,
                grid_dim=ceildiv(kv_dim * C, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
            )
            seed += 1

            # c_proj: zeros
            ctx.enqueue_function[fill_zeros_kernel, fill_zeros_kernel](
                self.c_proj[i], C * C,
                grid_dim=ceildiv(C * C, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
            )

            # c_fc: uniform(-s, s)
            ctx.enqueue_function[fill_uniform_kernel, fill_uniform_kernel](
                self.c_fc[i], seed, -s, s, mlp_dim * C,
                grid_dim=ceildiv(mlp_dim * C, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
            )
            seed += 1

            # mlp_proj: zeros
            ctx.enqueue_function[fill_zeros_kernel, fill_zeros_kernel](
                self.mlp_proj[i], C * mlp_dim,
                grid_dim=ceildiv(C * mlp_dim, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
            )

            # value_embeds: uniform(-s, s) if has_ve
            if self.config.has_ve(i):
                ctx.enqueue_function[fill_uniform_kernel, fill_uniform_kernel](
                    self.value_embeds[i], seed, -s, s, V * kv_dim,
                    grid_dim=ceildiv(V * kv_dim, LAUNCH_BLOCK), block_dim=LAUNCH_BLOCK,
                )
                seed += 1

                # ve_gate: zeros
                ctx.enqueue_function[fill_zeros_kernel, fill_zeros_kernel](
                    self.ve_gate[i], self.config.n_kv_head * VE_GATE_CHANNELS,
                    grid_dim=1, block_dim=LAUNCH_BLOCK,
                )

        # Precompute RoPE
        precompute_rope(ctx, MAX_SEQ_LEN, self.config.head_dim, self.rope_cos, self.rope_sin)

        ctx.synchronize()

# ── ForwardBuffers ───────────────────────────────────────────────────────


struct ForwardBuffers:
    """Pre-allocated activation buffers for the forward pass.

    All buffers are flat float32 arrays.
    """

    # Scratch buffers
    var x: DeviceBuffer[DType.float32]             # [B*T, C] current hidden state
    var x0: DeviceBuffer[DType.float32]            # [B*T, C] initial embedding (for skip connections)
    var x_norm: DeviceBuffer[DType.float32]        # [B*T, C] normalized hidden state
    var rms: DeviceBuffer[DType.float32]           # [B*T] saved RMS values
    var q: DeviceBuffer[DType.float32]             # [B*T, C] query projections
    var k: DeviceBuffer[DType.float32]             # [B*T, kv_dim] key projections
    var v: DeviceBuffer[DType.float32]             # [B*T, kv_dim] value projections
    var lse: DeviceBuffer[DType.float32]           # [B*T*n_head] log-sum-exp from flash attn
    var d_buf: DeviceBuffer[DType.float32]         # [B*T*n_head] dot(O, dO) for flash attn bwd
    var attn_out: DeviceBuffer[DType.float32]      # [B*T*n_head, head_dim] attention output
    var proj_out: DeviceBuffer[DType.float32]      # [B*T, C] after c_proj
    var mlp_hidden: DeviceBuffer[DType.float32]    # [B*T, 4*C] MLP hidden
    var mlp_act: DeviceBuffer[DType.float32]       # [B*T, 4*C] after relu squared
    var mlp_out: DeviceBuffer[DType.float32]       # [B*T, C] MLP output
    var logits: DeviceBuffer[DType.float32]        # [B*T, V] logits
    var loss: DeviceBuffer[DType.float32]          # [B*T] per-token loss
    var mean_loss: DeviceBuffer[DType.float32]     # [1] mean loss

    # Per-layer saved activations
    var saved_x: List[DeviceBuffer[DType.float32]]             # x before each block
    var saved_x_norm_attn: List[DeviceBuffer[DType.float32]]   # x_norm before attn
    var saved_q: List[DeviceBuffer[DType.float32]]             # q after RoPE+norm
    var saved_k: List[DeviceBuffer[DType.float32]]             # k after RoPE+norm
    var saved_v: List[DeviceBuffer[DType.float32]]             # v
    var saved_lse: List[DeviceBuffer[DType.float32]]           # lse from flash attn
    var saved_attn_out: List[DeviceBuffer[DType.float32]]      # attention output
    var saved_x_norm_mlp: List[DeviceBuffer[DType.float32]]    # x_norm before MLP
    var saved_mlp_hidden: List[DeviceBuffer[DType.float32]]    # before relu squared
    var saved_mlp_act: List[DeviceBuffer[DType.float32]]       # after relu squared
    var saved_rms_attn: List[DeviceBuffer[DType.float32]]      # RMS for attn norm
    var saved_rms_mlp: List[DeviceBuffer[DType.float32]]       # RMS for MLP norm
    var saved_x_pre_final: DeviceBuffer[DType.float32]         # x before final norm
    var saved_rms_final: DeviceBuffer[DType.float32]           # RMS for final norm
    var saved_x_final: DeviceBuffer[DType.float32]             # x after final norm (for lm_head bwd)

    # Gradient scratch
    var grad_x: DeviceBuffer[DType.float32]            # [B*T, C] gradient of x
    var grad_x0: DeviceBuffer[DType.float32]           # [B*T, C] gradient of x0
    var grad_logits: DeviceBuffer[DType.float32]       # [B*T, V]
    var grad_attn_out: DeviceBuffer[DType.float32]     # [B*T, C]
    var grad_q: DeviceBuffer[DType.float32]            # [B*T, C]
    var grad_k: DeviceBuffer[DType.float32]            # [B*T, kv_dim]
    var grad_v: DeviceBuffer[DType.float32]            # [B*T, kv_dim]
    var grad_x_norm: DeviceBuffer[DType.float32]       # [B*T, C]
    var grad_mlp_out: DeviceBuffer[DType.float32]      # [B*T, C]
    var grad_mlp_act: DeviceBuffer[DType.float32]      # [B*T, 4*C]
    var grad_mlp_hidden: DeviceBuffer[DType.float32]   # [B*T, 4*C]

    # VE scratch
    var ve_buf: DeviceBuffer[DType.float32]            # [B*T, kv_dim] value embedding lookup
    var ve_gate_buf: DeviceBuffer[DType.float32]       # [B*T, n_kv_head] gate values
    var grad_gate: DeviceBuffer[DType.float32]         # [B*T, n_kv_head] gate gradient scratch
    var grad_ve: DeviceBuffer[DType.float32]           # [B*T, kv_dim] VE gradient scratch
    var saved_ve: List[DeviceBuffer[DType.float32]]    # per-layer saved VE lookup
    var saved_gate: List[DeviceBuffer[DType.float32]]  # per-layer saved gate values

    fn __init__(out self, ctx: DeviceContext, B: Int, T: Int, config: GPTConfig) raises:
        var BT = B * T
        var C = config.n_embd
        var V = config.vocab_size
        var kv_dim = config.kv_dim
        var mlp_dim = MLP_EXPANSION * C

        self.x = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.x0 = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.x_norm = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.rms = ctx.enqueue_create_buffer[DType.float32](BT)
        self.q = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.k = ctx.enqueue_create_buffer[DType.float32](BT * kv_dim)
        self.v = ctx.enqueue_create_buffer[DType.float32](BT * kv_dim)
        var n_head = config.n_head
        self.lse = ctx.enqueue_create_buffer[DType.float32](BT * n_head)
        self.d_buf = ctx.enqueue_create_buffer[DType.float32](BT * n_head)
        self.attn_out = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.proj_out = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.mlp_hidden = ctx.enqueue_create_buffer[DType.float32](BT * mlp_dim)
        self.mlp_act = ctx.enqueue_create_buffer[DType.float32](BT * mlp_dim)
        self.mlp_out = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.logits = ctx.enqueue_create_buffer[DType.float32](BT * V)
        self.loss = ctx.enqueue_create_buffer[DType.float32](BT)
        self.mean_loss = ctx.enqueue_create_buffer[DType.float32](1)

        var L = config.n_layer
        self.saved_x = List[DeviceBuffer[DType.float32]]()
        self.saved_x_norm_attn = List[DeviceBuffer[DType.float32]]()
        self.saved_q = List[DeviceBuffer[DType.float32]]()
        self.saved_k = List[DeviceBuffer[DType.float32]]()
        self.saved_v = List[DeviceBuffer[DType.float32]]()
        self.saved_lse = List[DeviceBuffer[DType.float32]]()
        self.saved_attn_out = List[DeviceBuffer[DType.float32]]()
        self.saved_x_norm_mlp = List[DeviceBuffer[DType.float32]]()
        self.saved_mlp_hidden = List[DeviceBuffer[DType.float32]]()
        self.saved_mlp_act = List[DeviceBuffer[DType.float32]]()
        self.saved_rms_attn = List[DeviceBuffer[DType.float32]]()
        self.saved_rms_mlp = List[DeviceBuffer[DType.float32]]()
        for _ in range(L):
            self.saved_x.append(ctx.enqueue_create_buffer[DType.float32](BT * C))
            self.saved_x_norm_attn.append(ctx.enqueue_create_buffer[DType.float32](BT * C))
            self.saved_q.append(ctx.enqueue_create_buffer[DType.float32](BT * C))
            self.saved_k.append(ctx.enqueue_create_buffer[DType.float32](BT * kv_dim))
            self.saved_v.append(ctx.enqueue_create_buffer[DType.float32](BT * kv_dim))
            self.saved_lse.append(ctx.enqueue_create_buffer[DType.float32](BT * n_head))
            self.saved_attn_out.append(ctx.enqueue_create_buffer[DType.float32](BT * C))
            self.saved_x_norm_mlp.append(ctx.enqueue_create_buffer[DType.float32](BT * C))
            self.saved_mlp_hidden.append(ctx.enqueue_create_buffer[DType.float32](BT * mlp_dim))
            self.saved_mlp_act.append(ctx.enqueue_create_buffer[DType.float32](BT * mlp_dim))
            self.saved_rms_attn.append(ctx.enqueue_create_buffer[DType.float32](BT))
            self.saved_rms_mlp.append(ctx.enqueue_create_buffer[DType.float32](BT))
        self.saved_x_pre_final = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.saved_rms_final = ctx.enqueue_create_buffer[DType.float32](BT)
        self.saved_x_final = ctx.enqueue_create_buffer[DType.float32](BT * C)

        self.grad_x = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.grad_x0 = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.grad_logits = ctx.enqueue_create_buffer[DType.float32](BT * V)
        self.grad_attn_out = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.grad_q = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.grad_k = ctx.enqueue_create_buffer[DType.float32](BT * kv_dim)
        self.grad_v = ctx.enqueue_create_buffer[DType.float32](BT * kv_dim)
        self.grad_x_norm = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.grad_mlp_out = ctx.enqueue_create_buffer[DType.float32](BT * C)
        self.grad_mlp_act = ctx.enqueue_create_buffer[DType.float32](BT * mlp_dim)
        self.grad_mlp_hidden = ctx.enqueue_create_buffer[DType.float32](BT * mlp_dim)

        self.ve_buf = ctx.enqueue_create_buffer[DType.float32](BT * kv_dim)
        self.ve_gate_buf = ctx.enqueue_create_buffer[DType.float32](BT * n_head)
        self.grad_gate = ctx.enqueue_create_buffer[DType.float32](BT * n_head)
        self.grad_ve = ctx.enqueue_create_buffer[DType.float32](BT * kv_dim)
        self.saved_ve = List[DeviceBuffer[DType.float32]]()
        self.saved_gate = List[DeviceBuffer[DType.float32]]()
        for _ in range(L):
            self.saved_ve.append(ctx.enqueue_create_buffer[DType.float32](BT * kv_dim))
            self.saved_gate.append(ctx.enqueue_create_buffer[DType.float32](BT * n_head))

# ── Forward Pass ─────────────────────────────────────────────────────────


fn forward(
    ctx: DeviceContext,
    weights: ModelWeights,
    bufs: ForwardBuffers,
    input_ids: DeviceBuffer[DType.int64],
    targets: DeviceBuffer[DType.int64],
    B: Int,
    T: Int,
) raises -> Float32:
    """Run forward pass, compute cross-entropy loss. Returns mean loss.

    input_ids: [B*T] token indices
    targets: [B*T] target token indices

    Follows train.py's GPT.forward():
    1. Embedding lookup + RMSNorm
    2. For each layer: learned residual, pre-norm attention, pre-norm MLP
    3. Final norm, lm_head projection, softcap, cross-entropy
    """
    var config = weights.config
    var C = config.n_embd
    var V = config.vocab_size
    var BT = B * T
    var n_head = config.n_head
    var head_dim = config.head_dim
    var kv_dim = config.kv_dim
    var mlp_dim = MLP_EXPANSION * C
    comptime BLK = LAUNCH_BLOCK

    # --- 1. Embedding: x = wte[input_ids], shape [BT, C] ---
    ctx.enqueue_function[embedding_fwd, embedding_fwd](
        bufs.x, weights.wte, input_ids, BT, C,
        grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
    )

    # --- 2. RMSNorm on embedding: x = norm(x) ---
    ctx.enqueue_function[rmsnorm_fwd, rmsnorm_fwd](
        bufs.x, bufs.rms, bufs.x, BT, C,
        grid_dim=BT, block_dim=BLK,
    )

    # --- 3. Save x0 = x for skip connections ---
    ctx.enqueue_copy(dst_buf=bufs.x0, src_buf=bufs.x)

    # --- 4. Transformer blocks ---
    for layer_idx in range(config.n_layer):
        var ws = config.window_size(layer_idx)

        # 4a. Learned residual: x = resid_lambdas[i]*x + x0_lambdas[i]*x0
        ctx.enqueue_function[scaled_add_fwd, scaled_add_fwd](
            bufs.x, bufs.x, bufs.x0,
            weights.resid_lambdas, weights.x0_lambdas,
            layer_idx, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )

        ctx.enqueue_copy(dst_buf=bufs.saved_x[layer_idx], src_buf=bufs.x)

        # 4b. Pre-norm for attention: x_norm = norm(x)
        # Write directly to saved buffer, then copy to scratch
        ctx.enqueue_function[rmsnorm_fwd, rmsnorm_fwd](
            bufs.saved_x_norm_attn[layer_idx], bufs.saved_rms_attn[layer_idx], bufs.x, BT, C,
            grid_dim=BT, block_dim=BLK,
        )
        ctx.enqueue_copy(dst_buf=bufs.x_norm, src_buf=bufs.saved_x_norm_attn[layer_idx])

        # 4c. QKV projections: q = x_norm @ W_q^T, k = x_norm @ W_k^T, v = x_norm @ W_v^T
        launch_linear(ctx, bufs.q, bufs.x_norm, weights.c_q[layer_idx], BT, C, C)
        launch_linear(ctx, bufs.k, bufs.x_norm, weights.c_k[layer_idx], BT, kv_dim, C)
        launch_linear(ctx, bufs.v, bufs.x_norm, weights.c_v[layer_idx], BT, kv_dim, C)

        # 4d. Value embeddings
        if config.has_ve(layer_idx):
            ctx.enqueue_function[embedding_fwd, embedding_fwd](
                bufs.ve_buf, weights.value_embeds[layer_idx], input_ids, BT, kv_dim,
                grid_dim=ceildiv(BT * kv_dim, BLK), block_dim=BLK,
            )
            # gate = 2 * sigmoid(ve_gate @ x_norm[:, :, :32])
            ctx.enqueue_function[ve_gate_fwd, ve_gate_fwd](
                bufs.ve_gate_buf, bufs.x_norm, weights.ve_gate[layer_idx],
                BT, n_head, VE_GATE_CHANNELS, C,
                grid_dim=ceildiv(BT * n_head, BLK), block_dim=BLK,
            )
            ctx.enqueue_copy(dst_buf=bufs.saved_ve[layer_idx], src_buf=bufs.ve_buf)
            ctx.enqueue_copy(dst_buf=bufs.saved_gate[layer_idx], src_buf=bufs.ve_gate_buf)
            # v = v + gate * ve
            ctx.enqueue_function[ve_apply_fwd, ve_apply_fwd](
                bufs.v, bufs.ve_gate_buf, bufs.ve_buf, BT, n_head, head_dim,
                grid_dim=ceildiv(BT * n_head * head_dim, BLK), block_dim=BLK,
            )

        # 4e. RoPE on Q and K
        ctx.enqueue_function[rope_fwd, rope_fwd](
            bufs.q, bufs.q, weights.rope_cos, weights.rope_sin,
            BT, n_head, head_dim,
            grid_dim=ceildiv(BT * n_head * head_dim // 2, BLK), block_dim=BLK,
        )
        ctx.enqueue_function[rope_fwd, rope_fwd](
            bufs.k, bufs.k, weights.rope_cos, weights.rope_sin,
            BT, n_head, head_dim,
            grid_dim=ceildiv(BT * n_head * head_dim // 2, BLK), block_dim=BLK,
        )

        # 4f. RMSNorm on Q and K (per-head normalization)
        ctx.enqueue_function[rmsnorm_fwd, rmsnorm_fwd](
            bufs.q, bufs.rms, bufs.q, BT * n_head, head_dim,
            grid_dim=BT * n_head, block_dim=BLK,
        )
        ctx.enqueue_function[rmsnorm_fwd, rmsnorm_fwd](
            bufs.k, bufs.rms, bufs.k, BT * n_head, head_dim,
            grid_dim=BT * n_head, block_dim=BLK,
        )

        ctx.enqueue_copy(dst_buf=bufs.saved_q[layer_idx], src_buf=bufs.q)
        ctx.enqueue_copy(dst_buf=bufs.saved_k[layer_idx], src_buf=bufs.k)
        ctx.enqueue_copy(dst_buf=bufs.saved_v[layer_idx], src_buf=bufs.v)

        # 4g. Flash Attention: fused QK^T/sqrt(d) -> softmax -> @V, O(T) memory
        var bth_total = B * T * n_head
        ctx.enqueue_function[flash_attn_fwd, flash_attn_fwd](
            bufs.attn_out, bufs.lse, bufs.q, bufs.k, bufs.v,
            B, T, n_head, head_dim, ws,
            grid_dim=ceildiv(bth_total, BLK), block_dim=BLK,
        )
        ctx.enqueue_copy(dst_buf=bufs.saved_lse[layer_idx], src_buf=bufs.lse)
        ctx.enqueue_copy(dst_buf=bufs.saved_attn_out[layer_idx], src_buf=bufs.attn_out)

        # 4h. Output projection: proj_out = attn_out @ c_proj^T
        launch_linear(ctx, bufs.proj_out, bufs.attn_out, weights.c_proj[layer_idx], BT, C, C)

        # 4i. Residual: x = x + proj_out
        ctx.enqueue_function[add_residual_fwd, add_residual_fwd](
            bufs.x, bufs.x, bufs.proj_out, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )

        # 4j. Pre-norm for MLP: x_norm = norm(x)
        ctx.enqueue_function[rmsnorm_fwd, rmsnorm_fwd](
            bufs.x_norm, bufs.saved_rms_mlp[layer_idx], bufs.x, BT, C,
            grid_dim=BT, block_dim=BLK,
        )
        ctx.enqueue_copy(dst_buf=bufs.saved_x_norm_mlp[layer_idx], src_buf=bufs.x_norm)

        # 4k. MLP: hidden = c_fc(x_norm), act = relu_squared(hidden), out = mlp_proj(act)
        launch_linear(ctx, bufs.mlp_hidden, bufs.x_norm, weights.c_fc[layer_idx], BT, mlp_dim, C)
        ctx.enqueue_copy(dst_buf=bufs.saved_mlp_hidden[layer_idx], src_buf=bufs.mlp_hidden)

        ctx.enqueue_function[relu_squared_fwd, relu_squared_fwd](
            bufs.mlp_act, bufs.mlp_hidden, BT * mlp_dim,
            grid_dim=ceildiv(BT * mlp_dim, BLK), block_dim=BLK,
        )
        ctx.enqueue_copy(dst_buf=bufs.saved_mlp_act[layer_idx], src_buf=bufs.mlp_act)

        launch_linear(ctx, bufs.mlp_out, bufs.mlp_act, weights.mlp_proj[layer_idx], BT, C, mlp_dim)

        # 4l. Residual: x = x + mlp_out
        ctx.enqueue_function[add_residual_fwd, add_residual_fwd](
            bufs.x, bufs.x, bufs.mlp_out, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )

    # --- 5. Final RMSNorm ---
    ctx.enqueue_copy(dst_buf=bufs.saved_x_pre_final, src_buf=bufs.x)
    ctx.enqueue_function[rmsnorm_fwd, rmsnorm_fwd](
        bufs.x, bufs.saved_rms_final, bufs.x, BT, C,
        grid_dim=BT, block_dim=BLK,
    )
    ctx.enqueue_copy(dst_buf=bufs.saved_x_final, src_buf=bufs.x)

    # --- 6. LM head: logits = x @ lm_head^T, shape [BT, V] ---
    launch_linear(ctx, bufs.logits, bufs.x, weights.lm_head, BT, V, C)

    # --- 7. Softcap: logits = 15 * tanh(logits / 15) ---
    ctx.enqueue_function[softcap_fwd, softcap_fwd](
        bufs.logits, bufs.logits, Float32(SOFTCAP), BT * V,
        grid_dim=ceildiv(BT * V, BLK), block_dim=BLK,
    )

    # --- 8. Cross-entropy loss ---
    ctx.enqueue_function[cross_entropy_fwd, cross_entropy_fwd](
        bufs.loss, bufs.logits, targets, BT, V,
        grid_dim=BT, block_dim=BLK,
    )

    # --- 9. Mean loss ---
    ctx.enqueue_function[mean_reduce, mean_reduce](
        bufs.mean_loss, bufs.loss, BT,
        grid_dim=1, block_dim=BLK,
    )

    ctx.synchronize()

    var loss_host = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_buf=loss_host, src_buf=bufs.mean_loss)
    ctx.synchronize()

    return loss_host[0]

# ── Gradient Zeroing ─────────────────────────────────────────────────────


fn zero_grads(ctx: DeviceContext, weights: ModelWeights) raises:
    """Zero all parameter gradient buffers."""
    var config = weights.config
    var C = config.n_embd
    var V = config.vocab_size
    var L = config.n_layer
    var kv_dim = config.kv_dim
    var mlp_dim = MLP_EXPANSION * C
    comptime BLK = LAUNCH_BLOCK

    ctx.enqueue_function[zero_buffer, zero_buffer](
        weights.grad_wte, V * C, grid_dim=ceildiv(V * C, BLK), block_dim=BLK)
    ctx.enqueue_function[zero_buffer, zero_buffer](
        weights.grad_lm_head, V * C, grid_dim=ceildiv(V * C, BLK), block_dim=BLK)
    ctx.enqueue_function[zero_buffer, zero_buffer](
        weights.grad_resid_lambdas, L, grid_dim=1, block_dim=BLK)
    ctx.enqueue_function[zero_buffer, zero_buffer](
        weights.grad_x0_lambdas, L, grid_dim=1, block_dim=BLK)

    for i in range(L):
        ctx.enqueue_function[zero_buffer, zero_buffer](
            weights.grad_c_q[i], C * C, grid_dim=ceildiv(C * C, BLK), block_dim=BLK)
        ctx.enqueue_function[zero_buffer, zero_buffer](
            weights.grad_c_k[i], kv_dim * C, grid_dim=ceildiv(kv_dim * C, BLK), block_dim=BLK)
        ctx.enqueue_function[zero_buffer, zero_buffer](
            weights.grad_c_v[i], kv_dim * C, grid_dim=ceildiv(kv_dim * C, BLK), block_dim=BLK)
        ctx.enqueue_function[zero_buffer, zero_buffer](
            weights.grad_c_proj[i], C * C, grid_dim=ceildiv(C * C, BLK), block_dim=BLK)
        ctx.enqueue_function[zero_buffer, zero_buffer](
            weights.grad_c_fc[i], mlp_dim * C, grid_dim=ceildiv(mlp_dim * C, BLK), block_dim=BLK)
        ctx.enqueue_function[zero_buffer, zero_buffer](
            weights.grad_mlp_proj[i], C * mlp_dim, grid_dim=ceildiv(C * mlp_dim, BLK), block_dim=BLK)
        if config.has_ve(i):
            ctx.enqueue_function[zero_buffer, zero_buffer](
                weights.grad_value_embeds[i], V * kv_dim, grid_dim=ceildiv(V * kv_dim, BLK), block_dim=BLK)
            ctx.enqueue_function[zero_buffer, zero_buffer](
                weights.grad_ve_gate[i], config.n_kv_head * VE_GATE_CHANNELS,
                grid_dim=1, block_dim=BLK)

# ── Backward Pass ────────────────────────────────────────────────────────


fn backward(
    ctx: DeviceContext,
    weights: ModelWeights,
    bufs: ForwardBuffers,
    input_ids: DeviceBuffer[DType.int64],
    targets: DeviceBuffer[DType.int64],
    B: Int,
    T: Int,
    loss_scale: Float32,
) raises:
    """Backward pass: compute gradients for all parameters.

    Must be called after forward(). Uses saved activations from forward.
    Gradients are accumulated (not zeroed) -- call zero_grads() before.
    loss_scale: typically 1.0/grad_accum_steps for gradient accumulation.
    """
    var config = weights.config
    var C = config.n_embd
    var V = config.vocab_size
    var BT = B * T
    var n_head = config.n_head
    var head_dim = config.head_dim
    var kv_dim = config.kv_dim
    var mlp_dim = MLP_EXPANSION * C
    comptime BLK = LAUNCH_BLOCK

    # --- 1. Cross-entropy + softcap backward ---
    # grad_logits = (softmax(capped_logits) - one_hot) * loss_scale
    ctx.enqueue_function[cross_entropy_softcap_bwd, cross_entropy_softcap_bwd](
        bufs.grad_logits, bufs.logits, targets,
        Float32(SOFTCAP), BT, V, loss_scale,
        grid_dim=BT, block_dim=BLK,
    )

    # --- 2. LM head backward ---
    launch_linear_bwd_dx(ctx, bufs.grad_x, bufs.grad_logits, weights.lm_head, BT, V, C)
    launch_linear_bwd_dw(ctx, weights.grad_lm_head, bufs.grad_logits, bufs.saved_x_final, BT, V, C)

    # --- 3. Final RMSNorm backward ---
    ctx.enqueue_function[rmsnorm_bwd, rmsnorm_bwd](
        bufs.grad_x, bufs.grad_x, bufs.saved_x_pre_final,
        bufs.saved_rms_final, BT, C,
        grid_dim=BT, block_dim=BLK,
    )

    # --- 4. Zero grad_x0 (will accumulate from all layers) ---
    ctx.enqueue_function[zero_buffer, zero_buffer](
        bufs.grad_x0, BT * C, grid_dim=ceildiv(BT * C, BLK), block_dim=BLK)

    # --- 5. Transformer blocks (reverse order) ---
    for rev_idx in range(config.n_layer):
        var layer_idx = config.n_layer - 1 - rev_idx

        # 5l. Residual backward (MLP): grad_mlp_out = grad_x
        ctx.enqueue_copy(dst_buf=bufs.grad_mlp_out, src_buf=bufs.grad_x)

        # 5k. MLP backward
        launch_linear_bwd_dx(ctx, bufs.grad_mlp_act, bufs.grad_mlp_out, weights.mlp_proj[layer_idx], BT, C, mlp_dim)
        launch_linear_bwd_dw(ctx, weights.grad_mlp_proj[layer_idx], bufs.grad_mlp_out, bufs.saved_mlp_act[layer_idx], BT, C, mlp_dim)

        # relu squared backward
        ctx.enqueue_function[relu_squared_bwd, relu_squared_bwd](
            bufs.grad_mlp_hidden, bufs.grad_mlp_act,
            bufs.saved_mlp_hidden[layer_idx], BT * mlp_dim,
            grid_dim=ceildiv(BT * mlp_dim, BLK), block_dim=BLK,
        )

        # c_fc backward
        launch_linear_bwd_dx(ctx, bufs.grad_x_norm, bufs.grad_mlp_hidden, weights.c_fc[layer_idx], BT, mlp_dim, C)
        launch_linear_bwd_dw(ctx, weights.grad_c_fc[layer_idx], bufs.grad_mlp_hidden, bufs.saved_x_norm_mlp[layer_idx], BT, mlp_dim, C)

        # 5j. RMSNorm backward (MLP pre-norm)
        # x = x_before_mlp + mlp_out, so grad_x_before_mlp = grad_x + grad_through_mlp_norm
        ctx.enqueue_function[rmsnorm_bwd, rmsnorm_bwd](
            bufs.grad_x_norm, bufs.grad_x_norm, bufs.saved_x[layer_idx],
            bufs.saved_rms_mlp[layer_idx], BT, C,
            grid_dim=BT, block_dim=BLK,
        )
        ctx.enqueue_function[add_residual_fwd, add_residual_fwd](
            bufs.grad_x, bufs.grad_x, bufs.grad_x_norm, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )

        # 5i. Residual backward (attention): grad_attn_proj = grad_x
        ctx.enqueue_copy(dst_buf=bufs.grad_attn_out, src_buf=bufs.grad_x)

        # 5h. c_proj backward
        launch_linear_bwd_dx(ctx, bufs.grad_attn_out, bufs.grad_attn_out, weights.c_proj[layer_idx], BT, C, C)
        launch_linear_bwd_dw(ctx, weights.grad_c_proj[layer_idx], bufs.grad_x, bufs.saved_attn_out[layer_idx], BT, C, C)

        # 5g. Flash Attention backward (recomputes attention, O(T) memory)
        var bth_total = B * T * n_head
        var bthd_total = bth_total * head_dim
        var ws = config.window_size(layer_idx)

        # Step 0: D[bth] = sum_d O[bth,d] * dO[bth,d]
        ctx.enqueue_function[flash_attn_bwd_precompute_d, flash_attn_bwd_precompute_d](
            bufs.d_buf, bufs.saved_attn_out[layer_idx], bufs.grad_attn_out,
            B, T, n_head, head_dim,
            grid_dim=ceildiv(bth_total, BLK), block_dim=BLK,
        )
        # Step 1: dQ
        ctx.enqueue_function[flash_attn_bwd_dq, flash_attn_bwd_dq](
            bufs.grad_q,
            bufs.saved_q[layer_idx], bufs.saved_k[layer_idx], bufs.saved_v[layer_idx],
            bufs.grad_attn_out, bufs.saved_lse[layer_idx], bufs.d_buf,
            B, T, n_head, head_dim, ws,
            grid_dim=ceildiv(bthd_total, BLK), block_dim=BLK,
        )
        # Step 2: dK + dV
        ctx.enqueue_function[flash_attn_bwd_dkv, flash_attn_bwd_dkv](
            bufs.grad_k, bufs.grad_v,
            bufs.saved_q[layer_idx], bufs.saved_k[layer_idx], bufs.saved_v[layer_idx],
            bufs.grad_attn_out, bufs.saved_lse[layer_idx], bufs.d_buf,
            B, T, n_head, head_dim, ws,
            grid_dim=ceildiv(bthd_total, BLK), block_dim=BLK,
        )

        # 5f. Q/K RMSNorm backward -- skipped (norm is parameter-free, gradient passes through)

        # 5d. Value embedding backward: compute grad_ve and grad_gate from grad_v
        if config.has_ve(layer_idx):
            ctx.enqueue_function[ve_apply_bwd, ve_apply_bwd](
                bufs.grad_ve, bufs.grad_gate, bufs.grad_v,
                bufs.saved_gate[layer_idx], bufs.saved_ve[layer_idx],
                BT, n_head, head_dim,
                grid_dim=BT * n_head, block_dim=BLK,
            )
            # scatter_add grad_ve into grad_value_embeds
            ctx.enqueue_function[embedding_bwd, embedding_bwd](
                weights.grad_value_embeds[layer_idx], bufs.grad_ve,
                input_ids, BT, kv_dim,
                grid_dim=ceildiv(BT * kv_dim, BLK), block_dim=BLK,
            )

        # 5e. RoPE backward on Q and K
        ctx.enqueue_function[rope_bwd, rope_bwd](
            bufs.grad_q, bufs.grad_q, weights.rope_cos, weights.rope_sin,
            BT, n_head, head_dim,
            grid_dim=ceildiv(BT * n_head * head_dim // 2, BLK), block_dim=BLK,
        )
        ctx.enqueue_function[rope_bwd, rope_bwd](
            bufs.grad_k, bufs.grad_k, weights.rope_cos, weights.rope_sin,
            BT, n_head, head_dim,
            grid_dim=ceildiv(BT * n_head * head_dim // 2, BLK), block_dim=BLK,
        )

        # 5c. QKV linear backward
        launch_linear_bwd_dx(ctx, bufs.grad_x_norm, bufs.grad_q, weights.c_q[layer_idx], BT, C, C)
        launch_linear_bwd_dw(ctx, weights.grad_c_q[layer_idx], bufs.grad_q, bufs.saved_x_norm_attn[layer_idx], BT, C, C)

        # Add K contribution to grad_x_norm (reuse grad_mlp_out as scratch)
        launch_linear_bwd_dx(ctx, bufs.grad_mlp_out, bufs.grad_k, weights.c_k[layer_idx], BT, kv_dim, C)
        ctx.enqueue_function[add_residual_fwd, add_residual_fwd](
            bufs.grad_x_norm, bufs.grad_x_norm, bufs.grad_mlp_out, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )
        launch_linear_bwd_dw(ctx, weights.grad_c_k[layer_idx], bufs.grad_k, bufs.saved_x_norm_attn[layer_idx], BT, kv_dim, C)

        # Add V contribution to grad_x_norm
        launch_linear_bwd_dx(ctx, bufs.grad_mlp_out, bufs.grad_v, weights.c_v[layer_idx], BT, kv_dim, C)
        ctx.enqueue_function[add_residual_fwd, add_residual_fwd](
            bufs.grad_x_norm, bufs.grad_x_norm, bufs.grad_mlp_out, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )
        launch_linear_bwd_dw(ctx, weights.grad_c_v[layer_idx], bufs.grad_v, bufs.saved_x_norm_attn[layer_idx], BT, kv_dim, C)

        # 5c-ve. VE gate backward: accumulate into grad_x_norm AFTER QKV backward
        # gate = 2*sigmoid(W @ x[:,:32]) -> grad_ve_gate, grad_x_norm += contribution
        if config.has_ve(layer_idx):
            ctx.enqueue_function[ve_gate_bwd, ve_gate_bwd](
                bufs.grad_x_norm, weights.grad_ve_gate[layer_idx],
                bufs.grad_gate, bufs.saved_gate[layer_idx],
                bufs.saved_x_norm_attn[layer_idx], weights.ve_gate[layer_idx],
                BT, n_head, VE_GATE_CHANNELS, C,
                grid_dim=ceildiv(BT * n_head, BLK), block_dim=BLK,
            )

        # 5b. RMSNorm backward (attention pre-norm)
        ctx.enqueue_function[rmsnorm_bwd, rmsnorm_bwd](
            bufs.grad_x_norm, bufs.grad_x_norm, bufs.saved_x[layer_idx],
            bufs.saved_rms_attn[layer_idx], BT, C,
            grid_dim=BT, block_dim=BLK,
        )
        ctx.enqueue_function[add_residual_fwd, add_residual_fwd](
            bufs.grad_x, bufs.grad_x, bufs.grad_x_norm, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )

        # 5a. Scaled add backward: x = sa*x_prev + sb*x0
        # grad_x0 += sb * grad_x, grad_x = sa * grad_x
        ctx.enqueue_function[scalar_grad_reduce, scalar_grad_reduce](
            weights.grad_resid_lambdas, bufs.grad_x, bufs.saved_x[layer_idx],
            layer_idx, BT * C,
            grid_dim=1, block_dim=BLK,
        )
        ctx.enqueue_function[scalar_grad_reduce, scalar_grad_reduce](
            weights.grad_x0_lambdas, bufs.grad_x, bufs.x0,
            layer_idx, BT * C,
            grid_dim=1, block_dim=BLK,
        )
        ctx.enqueue_function[scaled_add_bwd, scaled_add_bwd](
            bufs.grad_x, bufs.grad_x0,
            bufs.grad_x,
            weights.resid_lambdas, weights.x0_lambdas,
            layer_idx, BT * C,
            grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
        )

    # --- 6. Embedding RMSNorm backward ---
    # The initial norm was applied in-place; we need the pre-norm embedding.
    # Since we saved x0 = norm(wte(idx)), and we need grad w.r.t. the embedding,
    # we skip the norm backward here (it's parameter-free).

    # --- 7. Embedding backward ---
    ctx.enqueue_function[embedding_bwd, embedding_bwd](
        weights.grad_wte, bufs.grad_x, input_ids, BT, C,
        grid_dim=ceildiv(BT * C, BLK), block_dim=BLK,
    )
