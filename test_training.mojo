# Phase 3: Test training loop with tiny model.
# Verifies loss decreases over multiple steps with AdamW.

from std.sys import has_accelerator
from std.math import ceildiv, log, isnan
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.python import Python, PythonObject
from std.time import perf_counter_ns

from config import GPTConfig, MLP_EXPANSION, ADAM_BETA1, ADAM_BETA2
from model import ModelWeights, ForwardBuffers, forward, backward, zero_grads
from optimizer import AdamWState


fn load_batch(
    ctx: DeviceContext,
    py_loader: PythonObject,
    input_buf: DeviceBuffer[DType.int64],
    target_buf: DeviceBuffer[DType.int64],
    x_host: HostBuffer[DType.int64],
    y_host: HostBuffer[DType.int64],
    BT: Int,
) raises -> Int:
    var np = Python.import_module("numpy")
    var batch = py_loader.__next__()
    var x_np = batch[0].numpy()
    var y_np = batch[1].numpy()
    var epoch = Int(py=batch[2])
    var x_ptr = x_np.ctypes.data.unsafe_get_as_pointer[DType.int64]()
    var y_ptr = y_np.ctypes.data.unsafe_get_as_pointer[DType.int64]()
    for i in range(BT):
        x_host[i] = x_ptr[i]
        y_host[i] = y_ptr[i]
    ctx.enqueue_copy(dst_buf=input_buf, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=target_buf, src_buf=y_host)
    return epoch


def main() raises:
    print("=== Phase 3: Training Loop Test ===")
    print()

    comptime if not has_accelerator():
        print("No GPU")
        return

    var ctx = DeviceContext()

    # Load tokenizer
    var prepare = Python.import_module("prepare")
    var py_tokenizer = prepare.Tokenizer.from_directory()
    var vocab_size = Int(py=py_tokenizer.get_vocab_size())
    print("Vocab size:", vocab_size)

    # Tiny config for fast testing
    var config = GPTConfig.default(vocab_size, 2)  # depth=2
    var C = config.n_embd
    var V = config.vocab_size
    var L = config.n_layer
    var kv_dim = config.kv_dim
    var mlp_dim = MLP_EXPANSION * C

    var B = 2   # tiny batch
    var T = 128  # short sequence
    var BT = B * T

    print("Config: L=", L, " C=", C, " V=", V, " B=", B, " T=", T)

    # Allocate model + buffers
    var weights = ModelWeights(ctx, config)
    weights.init_weights(ctx)
    var bufs = ForwardBuffers(ctx, B, T, config)

    # Setup optimizer
    var opt = AdamWState()
    var lr: Float64 = 0.001
    opt.add_param_group(ctx, V * C, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
    opt.add_param_group(ctx, V * C, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
    opt.add_param_group(ctx, L, lr=lr * 0.01, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
    opt.add_param_group(ctx, L, lr=lr, beta1=0.96, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
    for i in range(L):
        opt.add_param_group(ctx, C * C, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
        opt.add_param_group(ctx, kv_dim * C, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
        opt.add_param_group(ctx, kv_dim * C, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
        opt.add_param_group(ctx, C * C, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
        opt.add_param_group(ctx, mlp_dim * C, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)
        opt.add_param_group(ctx, C * mlp_dim, lr=lr, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=1e-10, weight_decay=0.0)

    # Dataloader
    var dl_cpu = Python.import_module("dataloader_cpu")
    var py_loader = dl_cpu.make_dataloader_cpu(py_tokenizer, B, T, "train")
    var input_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    var target_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    var x_host = ctx.enqueue_create_host_buffer[DType.int64](BT)
    var y_host = ctx.enqueue_create_host_buffer[DType.int64](BT)
    ctx.synchronize()

    # Prefetch
    _ = load_batch(ctx, py_loader, input_buf, target_buf, x_host, y_host, BT)
    ctx.synchronize()

    # Train for 10 steps
    var num_steps = 10
    print()
    print("Training for", num_steps, "steps...")

    var losses = List[Float64]()
    for step in range(num_steps):
        var t0 = perf_counter_ns()

        zero_grads(ctx, weights)
        ctx.synchronize()

        var loss = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
        backward(ctx, weights, bufs, input_buf, target_buf, B, T, Float32(1.0))
        ctx.synchronize()

        # Optimizer step
        opt.step += 1
        opt.step_group(ctx, 0, weights.lm_head, weights.grad_lm_head, 1.0)
        opt.step_group(ctx, 1, weights.wte, weights.grad_wte, 1.0)
        opt.step_group(ctx, 2, weights.resid_lambdas, weights.grad_resid_lambdas, 1.0)
        opt.step_group(ctx, 3, weights.x0_lambdas, weights.grad_x0_lambdas, 1.0)
        for i in range(L):
            var base = 4 + i * 6
            opt.step_group(ctx, base + 0, weights.c_q[i], weights.grad_c_q[i], 1.0)
            opt.step_group(ctx, base + 1, weights.c_k[i], weights.grad_c_k[i], 1.0)
            opt.step_group(ctx, base + 2, weights.c_v[i], weights.grad_c_v[i], 1.0)
            opt.step_group(ctx, base + 3, weights.c_proj[i], weights.grad_c_proj[i], 1.0)
            opt.step_group(ctx, base + 4, weights.c_fc[i], weights.grad_c_fc[i], 1.0)
            opt.step_group(ctx, base + 5, weights.mlp_proj[i], weights.grad_mlp_proj[i], 1.0)
        ctx.synchronize()

        _ = load_batch(ctx, py_loader, input_buf, target_buf, x_host, y_host, BT)
        ctx.synchronize()

        var t1 = perf_counter_ns()
        var dt_ms = Float64(t1 - t0) / 1e6

        var loss_f = Float64(loss)
        losses.append(loss_f)
        print("  step", step, "| loss:", loss_f, "| dt:", Int(dt_ms), "ms")

    # Verify loss decreased
    print()
    var pass_ = True

    if losses[num_steps - 1] >= losses[0]:
        print("FAIL: loss did not decrease (", losses[0], "->", losses[num_steps - 1], ")")
        pass_ = False
    else:
        print("OK: loss decreased from", losses[0], "to", losses[num_steps - 1])

    # Check for NaN
    for i in range(num_steps):
        if isnan(losses[i]):
            print("FAIL: NaN loss at step", i)
            pass_ = False

    print()
    if pass_:
        print("=== Training loop PASSED ===")
    else:
        print("=== Training loop FAILED ===")
