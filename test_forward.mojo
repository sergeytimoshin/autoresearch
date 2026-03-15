# Phase 1: Test forward pass end-to-end.
# Uses a tiny model to verify the forward pass produces correct loss values.

from std.sys import has_accelerator
from std.math import ceildiv, log, abs
from std.gpu.host import DeviceContext

from config import GPTConfig, MLP_EXPANSION
from model import ModelWeights, ForwardBuffers, forward, fill_uniform_kernel

comptime BLK = 256


def main() raises:
    print("=== Phase 1: Forward Pass Test ===")
    print()

    comptime if not has_accelerator():
        print("No GPU")
        return

    var ctx = DeviceContext()

    # Tiny model config for testing
    var config = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        head_dim=32,
        kv_dim=64,
    )
    var B = 2
    var T = 16
    var BT = B * T
    var C = config.n_embd
    var V = config.vocab_size

    print("Config: L=", config.n_layer, " C=", config.n_embd,
          " V=", config.vocab_size, " B=", B, " T=", T)

    # Allocate and initialize weights
    var weights = ModelWeights(ctx, config)
    weights.init_weights(ctx)

    # Also initialize c_proj and mlp_proj with small random values
    # (train.py zeros them, but we want to verify the forward path is functional)
    var seed: UInt64 = 12345
    var small_std: Float32 = 0.01
    var mlp_dim = MLP_EXPANSION * C
    for i in range(config.n_layer):
        ctx.enqueue_function[fill_uniform_kernel, fill_uniform_kernel](
            weights.c_proj[i], seed, -small_std, small_std, C * C,
            grid_dim=ceildiv(C * C, BLK), block_dim=BLK,
        )
        seed += 1
        ctx.enqueue_function[fill_uniform_kernel, fill_uniform_kernel](
            weights.mlp_proj[i], seed, -small_std, small_std, C * mlp_dim,
            grid_dim=ceildiv(C * mlp_dim, BLK), block_dim=BLK,
        )
        seed += 1
    ctx.synchronize()

    # Allocate forward buffers
    var bufs = ForwardBuffers(ctx, B, T, config)

    # Create input A: sequential tokens
    var inp_a = ctx.enqueue_create_host_buffer[DType.int64](BT)
    var tgt_a = ctx.enqueue_create_host_buffer[DType.int64](BT)
    ctx.synchronize()
    for i in range(BT):
        inp_a[i] = Int64(i % V)
        tgt_a[i] = Int64((i + 1) % V)

    var input_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    var target_buf = ctx.enqueue_create_buffer[DType.int64](BT)
    ctx.enqueue_copy(dst_buf=input_buf, src_buf=inp_a)
    ctx.enqueue_copy(dst_buf=target_buf, src_buf=tgt_a)
    ctx.synchronize()

    var loss_a = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
    print("Loss A (sequential):", loss_a)

    # Create input B: different pattern
    for i in range(BT):
        inp_a[i] = Int64((i * 7 + 3) % V)
        tgt_a[i] = Int64((i * 7 + 4) % V)
    ctx.enqueue_copy(dst_buf=input_buf, src_buf=inp_a)
    ctx.enqueue_copy(dst_buf=target_buf, src_buf=tgt_a)
    ctx.synchronize()

    var loss_b = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
    print("Loss B (scrambled):", loss_b)

    # Determinism: run A again
    for i in range(BT):
        inp_a[i] = Int64(i % V)
        tgt_a[i] = Int64((i + 1) % V)
    ctx.enqueue_copy(dst_buf=input_buf, src_buf=inp_a)
    ctx.enqueue_copy(dst_buf=target_buf, src_buf=tgt_a)
    ctx.synchronize()

    var loss_a2 = forward(ctx, weights, bufs, input_buf, target_buf, B, T)
    print("Loss A (repeat):", loss_a2)

    print()
    var pass_ = True
    var expected = log(Float32(V))
    print("Expected random loss: ~", expected)

    # Check losses are positive and reasonable
    if loss_a <= 0.0 or loss_b <= 0.0:
        print("FAIL: loss should be positive")
        pass_ = False

    if loss_a > expected * 3.0 or loss_b > expected * 3.0:
        print("FAIL: loss too high")
        pass_ = False

    # With non-zero projections, different inputs should give different losses
    if abs(loss_a - loss_b) < 1e-6:
        print("WARN: same loss for different inputs (model may not be differentiating)")
    else:
        print("OK: different inputs give different losses (diff=", abs(loss_a - loss_b), ")")

    # Determinism check
    if abs(loss_a - loss_a2) > 1e-4:
        print("FAIL: non-deterministic! loss_a=", loss_a, "loss_a2=", loss_a2)
        pass_ = False
    else:
        print("OK: deterministic (loss_a == loss_a2)")

    print()
    if pass_:
        print("=== Forward pass PASSED ===")
    else:
        print("=== Forward pass FAILED ===")
