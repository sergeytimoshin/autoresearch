# Single-GPU GPT training with manual forward/backward and AdamW + Muon.
#
# Usage:
#   pixi run mojo build -o train train.mojo
#   TRAIN_TIME_BUDGET=300 pixi run ./train

from std.sys import has_accelerator
from std.math import ceildiv, log, isnan
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.python import Python, PythonObject
from std.time import perf_counter_ns

from config import (
    GPTConfig, MAX_SEQ_LEN, DEPTH, TOTAL_BATCH_SIZE,
    EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR, WEIGHT_DECAY,
    ADAM_BETA1, ADAM_BETA2, WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC,
    TIME_BUDGET, MLP_EXPANSION, VE_GATE_CHANNELS,
)
from model import ModelWeights, ForwardBuffers, forward, backward, zero_grads
from optimizer import AdamWState, MuonState


# ── LR schedule ──────────────────────────────────────────────────────────────

fn get_lr_multiplier(progress: Float64) -> Float64:
    """Warmup → constant → cosine warmdown to FINAL_LR_FRAC."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        var cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


# ── Data loading ─────────────────────────────────────────────────────────────

fn load_batch(
    ctx: DeviceContext,
    py_loader: PythonObject,
    input_buf: DeviceBuffer[DType.int64],
    target_buf: DeviceBuffer[DType.int64],
    x_host: HostBuffer[DType.int64],
    y_host: HostBuffer[DType.int64],
    num_tokens: Int,
) raises -> Int:
    """Fetch next batch from Python, copy to GPU. Returns epoch number."""
    var np = Python.import_module("numpy")
    var batch = py_loader.__next__()
    var x_np = batch[0].numpy()
    var y_np = batch[1].numpy()
    var epoch = Int(py=batch[2])

    var x_ptr = x_np.ctypes.data.unsafe_get_as_pointer[DType.int64]()
    var y_ptr = y_np.ctypes.data.unsafe_get_as_pointer[DType.int64]()
    for i in range(num_tokens):
        x_host[i] = x_ptr[i]
        y_host[i] = y_ptr[i]

    ctx.enqueue_copy(dst_buf=input_buf, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=target_buf, src_buf=y_host)
    return epoch


# ── BPB evaluation ───────────────────────────────────────────────────────────

fn evaluate_bpb(
    ctx: DeviceContext,
    weights: ModelWeights,
    bufs: ForwardBuffers,
    input_buf: DeviceBuffer[DType.int64],
    target_buf: DeviceBuffer[DType.int64],
    x_host: HostBuffer[DType.int64],
    y_host: HostBuffer[DType.int64],
    py_tokenizer: PythonObject,
    prepare: PythonObject,
    dl_cpu: PythonObject,
    B: Int,
    T: Int,
) raises -> Float64:
    """Bits-per-byte on the validation set (lower is better)."""
    var BT = B * T
    var V = weights.config.vocab_size

    # Token byte lengths from prepare.py (int32 tensor mapping token_id → #bytes)
    var token_bytes_np = prepare.get_token_bytes().numpy()
    var token_bytes_ptr = token_bytes_np.ctypes.data.unsafe_get_as_pointer[DType.int32]()
    var token_bytes_host = ctx.enqueue_create_host_buffer[DType.int32](V)
    ctx.synchronize()
    for i in range(V):
        token_bytes_host[i] = token_bytes_ptr[i]

    var val_loader = dl_cpu.make_dataloader_cpu(py_tokenizer, B, T, "val")

    # Cap eval steps for small batches
    var eval_steps = 40 * 524288 // BT
    if eval_steps > 100:
        eval_steps = 100

    var total_nats: Float64 = 0.0
    var total_bytes: Float64 = 0.0
    var loss_host = ctx.enqueue_create_host_buffer[DType.float32](BT)
    ctx.synchronize()

    for _ in range(eval_steps):
        _ = load_batch(ctx, val_loader, input_buf, target_buf, x_host, y_host, BT)
        ctx.synchronize()
        _ = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
        ctx.enqueue_copy(dst_buf=loss_host, src_buf=bufs.loss)
        ctx.synchronize()

        # Accumulate nats weighted by byte length (skip special tokens with 0 bytes)
        for i in range(BT):
            var target_id = Int(y_host[i])
            if target_id >= 0 and target_id < V:
                var nbytes = Int(token_bytes_host[target_id])
                if nbytes > 0:
                    total_nats += Float64(loss_host[i])
                    total_bytes += Float64(nbytes)

    if total_bytes > 0:
        return total_nats / (0.6931471805599453 * total_bytes)  # nats → bits
    return 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() raises:
    print("=== Autoresearch Mojo Training ===")
    print()

    comptime if not has_accelerator():
        print("ERROR: No compatible GPU found.")
        return

    var t_start = perf_counter_ns()
    var ctx = DeviceContext()

    # ── Tokenizer ──
    var prepare = Python.import_module("prepare")
    var py_tokenizer = prepare.Tokenizer.from_directory()
    var vocab_size = Int(py=py_tokenizer.get_vocab_size())

    # ── Config ──
    var config = GPTConfig.default(vocab_size, DEPTH)
    var C = config.n_embd
    var V = config.vocab_size
    var L = config.n_layer
    var kv_dim = config.kv_dim
    var mlp_dim = MLP_EXPANSION * C

    # Training batch dimensions
    var B = 4
    var T = 512
    var BT = B * T
    var grad_accum_steps = TOTAL_BATCH_SIZE // BT
    if grad_accum_steps < 1:
        grad_accum_steps = 1

    print("Model: L=", L, " H=", config.n_head, " C=", C, " V=", V)
    print("Batch: B=", B, " T=", T, " micro=", BT, " total=", BT * grad_accum_steps, " accum=", grad_accum_steps)

    # ── Model ──
    var weights = ModelWeights(ctx, config)
    weights.init_weights(ctx)
    var bufs = ForwardBuffers(ctx, B, T, config)

    # ── Optimizer ──
    # LR scaled by (C/768)^{-0.5} to normalize across model widths
    var lr_scale = (Float64(C) / 768.0) ** -0.5

    var opt = AdamWState()
    opt.add_param_group(ctx, V * C,                                       # lm_head
        lr=UNEMBEDDING_LR * lr_scale, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
    opt.add_param_group(ctx, V * C,                                       # wte
        lr=EMBEDDING_LR * lr_scale, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
    opt.add_param_group(ctx, L,                                           # resid_lambdas
        lr=SCALAR_LR * 0.01, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
    opt.add_param_group(ctx, L,                                           # x0_lambdas
        lr=SCALAR_LR, beta1=0.96, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)

    # VE params: value_embeds + ve_gate per layer (AdamW)
    var ve_group_start = 4
    for i in range(L):
        if config.has_ve(i):
            opt.add_param_group(ctx, V * kv_dim, lr=EMBEDDING_LR * lr_scale,
                beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
            opt.add_param_group(ctx, config.n_kv_head * VE_GATE_CHANNELS, lr=MATRIX_LR,
                beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)

    # Matrix params: Muon (Polar Express orthogonalization)
    var muon = MuonState(ctx, initial_lr=MATRIX_LR)
    for _ in range(L):
        muon.add_param(ctx, C, C)           # c_q
        muon.add_param(ctx, kv_dim, C)      # c_k
        muon.add_param(ctx, kv_dim, C)      # c_v
        muon.add_param(ctx, C, C)           # c_proj
        muon.add_param(ctx, mlp_dim, C)     # c_fc
        muon.add_param(ctx, C, mlp_dim)     # mlp_proj

    # ── Dataloader ──
    var dl_cpu = Python.import_module("dataloader_cpu")
    var py_loader = dl_cpu.make_dataloader_cpu(py_tokenizer, B, T, "train")

    var input_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    var target_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    var x_host = ctx.enqueue_create_host_buffer[DType.int64](BT)
    var y_host = ctx.enqueue_create_host_buffer[DType.int64](BT)
    ctx.synchronize()

    var epoch = load_batch(ctx, py_loader, input_buf, target_buf, x_host, y_host, BT)
    ctx.synchronize()

    # ── Time budget (overridable via TRAIN_TIME_BUDGET env var) ──
    var time_budget = TIME_BUDGET
    try:
        var os = Python.import_module("os")
        var budget_str = os.environ.get("TRAIN_TIME_BUDGET", "")
        if String(py=budget_str) != "":
            time_budget = Int(py=budget_str)
    except:
        pass

    print("Time budget:", time_budget, "s")
    print()

    # ── Training loop ────────────────────────────────────────────────────────

    var t_train_start = perf_counter_ns()
    var training_time: Float64 = 0.0
    var smooth_loss: Float64 = 0.0
    var step = 0

    while True:
        var t0 = perf_counter_ns()

        # Zero gradients once per optimizer step
        zero_grads(ctx, weights)
        ctx.synchronize()

        # Gradient accumulation: multiple forward+backward micro-batches
        var acc_loss: Float64 = 0.0
        var scale = Float32(1.0 / Float64(grad_accum_steps))
        for _micro in range(grad_accum_steps):
            var loss = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
            backward(ctx, weights, bufs, input_buf, target_buf, B, T, scale)
            ctx.synchronize()
            acc_loss += Float64(loss) / Float64(grad_accum_steps)

            # Load next micro-batch
            epoch = load_batch(ctx, py_loader, input_buf, target_buf, x_host, y_host, BT)
            ctx.synchronize()

        var loss_f = acc_loss

        # LR schedule
        var progress = training_time / Float64(time_budget)
        if progress > 1.0:
            progress = 1.0
        var lrm = get_lr_multiplier(progress)

        # AdamW step (embeddings + scalars)
        opt.step += 1
        opt.step_group(ctx, 0, weights.lm_head, weights.grad_lm_head, lrm)
        opt.step_group(ctx, 1, weights.wte, weights.grad_wte, lrm)
        opt.step_group_f32(ctx, 2, weights.resid_lambdas, weights.grad_resid_lambdas, lrm)
        opt.step_group_f32(ctx, 3, weights.x0_lambdas, weights.grad_x0_lambdas, lrm)

        # AdamW step (VE params)
        var ve_gi = ve_group_start
        for i in range(L):
            if config.has_ve(i):
                opt.step_group(ctx, ve_gi, weights.value_embeds[i], weights.grad_value_embeds[i], lrm)
                opt.step_group_f32(ctx, ve_gi + 1, weights.ve_gate[i], weights.grad_ve_gate[i], lrm)
                ve_gi += 2

        # Muon step (matrix params): momentum ramps 0.85 → 0.95 over first 300 steps
        var mu_frac = Float64(opt.step) / 300.0
        if mu_frac > 1.0:
            mu_frac = 1.0
        var mu_momentum = (1.0 - mu_frac) * 0.85 + mu_frac * 0.95
        var mu_wd = WEIGHT_DECAY * (1.0 - progress)

        for i in range(L):
            var base = i * 6
            muon.step(ctx, base + 0, weights.c_q[i], weights.grad_c_q[i], lrm, mu_momentum, mu_wd)
            muon.step(ctx, base + 1, weights.c_k[i], weights.grad_c_k[i], lrm, mu_momentum, mu_wd)
            muon.step(ctx, base + 2, weights.c_v[i], weights.grad_c_v[i], lrm, mu_momentum, mu_wd)
            muon.step(ctx, base + 3, weights.c_proj[i], weights.grad_c_proj[i], lrm, mu_momentum, mu_wd)
            muon.step(ctx, base + 4, weights.c_fc[i], weights.grad_c_fc[i], lrm, mu_momentum, mu_wd)
            muon.step(ctx, base + 5, weights.mlp_proj[i], weights.grad_mlp_proj[i], lrm, mu_momentum, mu_wd)
        ctx.synchronize()

        # Fast fail on NaN / explosion
        if isnan(loss_f) or loss_f > 100.0:
            print("FAIL: loss =", loss_f)
            return

        # Timing (skip warmup steps 0-10)
        var t1 = perf_counter_ns()
        var dt = Float64(t1 - t0) / 1e9
        if step > 10:
            training_time += dt

        # Log
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
        var debiased = smooth_loss / (1.0 - 0.9 ** Float64(step + 1))
        var remaining = Float64(time_budget) - training_time
        if remaining < 0:
            remaining = 0

        print(
            "step", step,
            " |", Int(100.0 * progress), "% |",
            " loss:", debiased,
            " | lrm:", lrm,
            " | dt:", Int(dt * 1000), "ms",
            " | epoch:", epoch,
            " | rem:", Int(remaining), "s",
        )

        step += 1
        if step > 10 and training_time >= Float64(time_budget):
            break

    # ── BPB evaluation ───────────────────────────────────────────────────────

    print()
    print("Evaluating validation BPB...")
    var val_bpb = evaluate_bpb(
        ctx, weights, bufs, input_buf, target_buf, x_host, y_host,
        py_tokenizer, prepare, dl_cpu, B, T,
    )

    # ── Summary ──────────────────────────────────────────────────────────────

    var t_end = perf_counter_ns()
    var startup_s = Float64(t_train_start - t_start) / 1e9
    var total_s = Float64(t_end - t_start) / 1e9

    print()
    print("---")
    print("val_bpb:         ", val_bpb)
    print("training_seconds:", training_time)
    print("total_seconds:   ", total_s)
    print("startup_seconds: ", startup_s)
    print("total_tokens_M:  ", Float64(step * BT * grad_accum_steps) / 1e6)
    print("num_steps:       ", step)
    print("depth:           ", DEPTH)
