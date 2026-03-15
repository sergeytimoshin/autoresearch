# Phase 2: Test backward pass.
# Verifies gradients are non-zero and a single SGD step reduces loss.

from std.sys import has_accelerator
from std.math import ceildiv, abs
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import block_dim, block_idx, thread_idx

from config import GPTConfig, MLP_EXPANSION
from model import ModelWeights, ForwardBuffers, forward, backward, zero_grads

comptime BLK = 256


# Simple SGD kernel: param -= lr * grad
def sgd_step_kernel(
    params: UnsafePointer[Float32, MutAnyOrigin],
    grads: UnsafePointer[Float32, MutAnyOrigin],
    lr: Float32,
    size: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= size:
        return
    params[tid] = params[tid] - lr * grads[tid]


fn sgd_update(
    ctx: DeviceContext,
    params: DeviceBuffer[DType.float32],
    grads: DeviceBuffer[DType.float32],
    lr: Float32,
    size: Int,
) raises:
    ctx.enqueue_function[sgd_step_kernel, sgd_step_kernel](
        params, grads, lr, size,
        grid_dim=ceildiv(size, BLK), block_dim=BLK,
    )


def main() raises:
    print("=== Phase 2: Backward Pass Test ===")
    print()

    comptime if not has_accelerator():
        print("No GPU")
        return

    var ctx = DeviceContext()

    # Tiny model
    var config = GPTConfig(
        sequence_len=8,
        vocab_size=16,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        head_dim=32,
        kv_dim=64,
    )
    var B = 2
    var T = 8
    var BT = B * T
    var C = config.n_embd
    var V = config.vocab_size

    print("Config: L=", config.n_layer, " C=", C, " V=", V, " B=", B, " T=", T)

    # Allocate model
    var weights = ModelWeights(ctx, config)
    weights.init_weights(ctx)
    var bufs = ForwardBuffers(ctx, B, T, config)

    # Create dummy input
    var input_host = ctx.enqueue_create_host_buffer[DType.int64](BT)
    var target_host = ctx.enqueue_create_host_buffer[DType.int64](BT)
    ctx.synchronize()
    for i in range(BT):
        input_host[i] = Int64(i % V)
        target_host[i] = Int64((i + 1) % V)
    var input_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    var target_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    ctx.enqueue_copy(dst_buf=input_buf, src_buf=input_host)
    ctx.enqueue_copy(dst_buf=target_buf, src_buf=target_host)
    ctx.synchronize()

    # Forward
    var loss0 = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
    print("Initial loss:", loss0)

    # Backward
    zero_grads(ctx, weights)
    ctx.synchronize()
    backward(ctx, weights, bufs, input_buf, target_buf, B, T, Float32(1.0))
    ctx.synchronize()

    # Check some gradients are non-zero
    var grad_host = ctx.enqueue_create_host_buffer[DType.float32](V * C)
    ctx.enqueue_copy(dst_buf=grad_host, src_buf=weights.grad_lm_head)
    ctx.synchronize()

    var nonzero_count = 0
    var grad_norm: Float32 = 0.0
    for i in range(V * C):
        if abs(grad_host[i]) > 1e-10:
            nonzero_count += 1
        grad_norm += grad_host[i] * grad_host[i]

    print("lm_head grad: nonzero=", nonzero_count, "/", V * C, " norm²=", grad_norm)

    var pass_ = True
    if nonzero_count == 0:
        print("  FAIL: all lm_head gradients are zero!")
        pass_ = False
    else:
        print("  OK: gradients are non-zero")

    # Check wte gradients
    ctx.enqueue_copy(dst_buf=grad_host, src_buf=weights.grad_wte)
    ctx.synchronize()
    nonzero_count = 0
    for i in range(V * C):
        if abs(grad_host[i]) > 1e-10:
            nonzero_count += 1
    print("wte grad: nonzero=", nonzero_count, "/", V * C)
    if nonzero_count == 0:
        print("  FAIL: all wte gradients are zero!")
        pass_ = False

    # SGD step and check loss decreases
    var lr: Float32 = 0.01
    var mlp_dim = MLP_EXPANSION * C
    var kv_dim = config.kv_dim

    sgd_update(ctx, weights.wte, weights.grad_wte, lr, V * C)
    sgd_update(ctx, weights.lm_head, weights.grad_lm_head, lr, V * C)
    for i in range(config.n_layer):
        sgd_update(ctx, weights.c_q[i], weights.grad_c_q[i], lr, C * C)
        sgd_update(ctx, weights.c_k[i], weights.grad_c_k[i], lr, kv_dim * C)
        sgd_update(ctx, weights.c_v[i], weights.grad_c_v[i], lr, kv_dim * C)
        sgd_update(ctx, weights.c_proj[i], weights.grad_c_proj[i], lr, C * C)
        sgd_update(ctx, weights.c_fc[i], weights.grad_c_fc[i], lr, mlp_dim * C)
        sgd_update(ctx, weights.mlp_proj[i], weights.grad_mlp_proj[i], lr, C * mlp_dim)
    ctx.synchronize()

    # Forward again with updated weights
    var loss1 = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
    print()
    print("Loss after 1 SGD step (lr=", lr, "):", loss1)
    print("Loss change:", loss1 - loss0)

    if loss1 < loss0:
        print("  OK: loss decreased!")
    else:
        print("  FAIL: loss did not decrease (", loss0, "->", loss1, ")")
        pass_ = False

    # Do a few more steps
    for step in range(9):
        zero_grads(ctx, weights)
        ctx.synchronize()
        _ = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
        backward(ctx, weights, bufs, input_buf, target_buf, B, T, Float32(1.0))
        ctx.synchronize()
        sgd_update(ctx, weights.wte, weights.grad_wte, lr, V * C)
        sgd_update(ctx, weights.lm_head, weights.grad_lm_head, lr, V * C)
        for i in range(config.n_layer):
            sgd_update(ctx, weights.c_q[i], weights.grad_c_q[i], lr, C * C)
            sgd_update(ctx, weights.c_k[i], weights.grad_c_k[i], lr, kv_dim * C)
            sgd_update(ctx, weights.c_v[i], weights.grad_c_v[i], lr, kv_dim * C)
            sgd_update(ctx, weights.c_proj[i], weights.grad_c_proj[i], lr, C * C)
            sgd_update(ctx, weights.c_fc[i], weights.grad_c_fc[i], lr, mlp_dim * C)
            sgd_update(ctx, weights.mlp_proj[i], weights.grad_mlp_proj[i], lr, C * mlp_dim)
        ctx.synchronize()

    var loss10 = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
    print("Loss after 10 SGD steps:", loss10)

    if loss10 < loss0:
        print("  OK: training is working (loss: ", loss0, " -> ", loss10, ")")
    else:
        print("  FAIL: loss not decreasing over 10 steps")
        pass_ = False

    print()
    if pass_:
        print("=== Backward pass PASSED ===")
    else:
        print("=== Backward pass FAILED ===")
