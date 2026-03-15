# Phase 1: Test each GPU kernel individually.
# Verifies correct output by comparing against known values.

from std.sys import has_accelerator
from std.math import ceildiv, abs
from std.gpu.host import DeviceContext, DeviceBuffer

from kernels.ops import (
    embedding_fwd,
    relu_squared_fwd,
    softcap_fwd,
    rmsnorm_fwd,
    softmax_fwd,
    cross_entropy_fwd,
)

comptime BLOCK_SIZE = 256


def test_embedding(ctx: DeviceContext) raises:
    """Test embedding lookup: output[i] = weight[idx[i]]."""
    print("--- Test: Embedding Forward ---")
    comptime vocab = 8
    comptime dim = 4
    comptime num_tokens = 3

    var w_host = ctx.enqueue_create_host_buffer[DType.float32](vocab * dim)
    ctx.synchronize()
    for v in range(vocab):
        for d in range(dim):
            w_host[v * dim + d] = Float32(v * 10 + d)

    var idx_host = ctx.enqueue_create_host_buffer[DType.int64](num_tokens)
    ctx.synchronize()
    idx_host[0] = 2; idx_host[1] = 5; idx_host[2] = 0

    var w_dev = ctx.enqueue_create_buffer[DType.float32](vocab * dim)
    var idx_dev = ctx.enqueue_create_buffer[DType.int64](num_tokens)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](num_tokens * dim)
    ctx.enqueue_copy(dst_buf=w_dev, src_buf=w_host)
    ctx.enqueue_copy(dst_buf=idx_dev, src_buf=idx_host)

    var total = num_tokens * dim
    ctx.enqueue_function[embedding_fwd, embedding_fwd](
        out_dev, w_dev, idx_dev, num_tokens, dim,
        grid_dim=ceildiv(total, BLOCK_SIZE), block_dim=BLOCK_SIZE,
    )

    var out_host = ctx.enqueue_create_host_buffer[DType.float32](num_tokens * dim)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    var exp_host = ctx.enqueue_create_host_buffer[DType.float32](num_tokens * dim)
    ctx.synchronize()
    exp_host[0] = 20; exp_host[1] = 21; exp_host[2] = 22; exp_host[3] = 23
    exp_host[4] = 50; exp_host[5] = 51; exp_host[6] = 52; exp_host[7] = 53
    exp_host[8] = 0; exp_host[9] = 1; exp_host[10] = 2; exp_host[11] = 3
    var pass_ = True
    for i in range(num_tokens * dim):
        if out_host[i] != exp_host[i]:
            print("  FAIL at", i, ": got", out_host[i], "expected", exp_host[i])
            pass_ = False
    if pass_:
        print("  PASS")


def test_relu_squared(ctx: DeviceContext) raises:
    """Test relu²: output = relu(x)² = max(0,x)²."""
    print("--- Test: ReLU² Forward ---")
    comptime size = 6

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](size)
    ctx.synchronize()
    in_host[0] = -2.0; in_host[1] = -1.0; in_host[2] = 0.0
    in_host[3] = 1.0; in_host[4] = 2.0; in_host[5] = 3.0

    var in_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(dst_buf=in_dev, src_buf=in_host)

    ctx.enqueue_function[relu_squared_fwd, relu_squared_fwd](
        out_dev, in_dev, size,
        grid_dim=1, block_dim=BLOCK_SIZE,
    )

    var out_host = ctx.enqueue_create_host_buffer[DType.float32](size)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    var exp_host = ctx.enqueue_create_host_buffer[DType.float32](size)
    ctx.synchronize()
    exp_host[0] = 0; exp_host[1] = 0; exp_host[2] = 0
    exp_host[3] = 1; exp_host[4] = 4; exp_host[5] = 9
    var pass_ = True
    for i in range(size):
        if abs(out_host[i] - exp_host[i]) > 1e-5:
            print("  FAIL at", i, ": got", out_host[i], "expected", exp_host[i])
            pass_ = False
    if pass_:
        print("  PASS")


def test_softcap(ctx: DeviceContext) raises:
    """Test softcap: output = cap * tanh(x / cap)."""
    print("--- Test: Softcap Forward ---")
    comptime size = 4

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](size)
    ctx.synchronize()
    in_host[0] = 0.0; in_host[1] = 15.0; in_host[2] = -30.0; in_host[3] = 100.0

    var in_dev = ctx.enqueue_create_buffer[DType.float32](size)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_copy(dst_buf=in_dev, src_buf=in_host)

    var cap: Float32 = 15.0
    ctx.enqueue_function[softcap_fwd, softcap_fwd](
        out_dev, in_dev, cap, size,
        grid_dim=1, block_dim=BLOCK_SIZE,
    )

    var out_host = ctx.enqueue_create_host_buffer[DType.float32](size)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    var pass_ = True
    if abs(out_host[0]) > 1e-5:
        print("  FAIL: softcap(0) =", out_host[0], "expected ~0")
        pass_ = False
    if abs(out_host[3] - 15.0) > 0.1:
        print("  FAIL: softcap(100) =", out_host[3], "expected ~15")
        pass_ = False
    if out_host[2] > -14.0:
        print("  FAIL: softcap(-30) =", out_host[2], "expected < -14")
        pass_ = False
    if pass_:
        print("  PASS (softcap(0)=", out_host[0], "softcap(100)=", out_host[3], ")")


def test_rmsnorm(ctx: DeviceContext) raises:
    """Test RMSNorm: y = x / sqrt(mean(x²) + eps)."""
    print("--- Test: RMSNorm Forward ---")
    comptime dim = 4
    comptime num_rows = 2

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](num_rows * dim)
    ctx.synchronize()
    in_host[0] = 1.0; in_host[1] = 2.0; in_host[2] = 3.0; in_host[3] = 4.0
    in_host[4] = 0.0; in_host[5] = 0.0; in_host[6] = 0.0; in_host[7] = 5.0

    var in_dev = ctx.enqueue_create_buffer[DType.float32](num_rows * dim)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](num_rows * dim)
    var rms_dev = ctx.enqueue_create_buffer[DType.float32](num_rows)
    ctx.enqueue_copy(dst_buf=in_dev, src_buf=in_host)

    ctx.enqueue_function[rmsnorm_fwd, rmsnorm_fwd](
        out_dev, rms_dev, in_dev, num_rows, dim,
        grid_dim=num_rows, block_dim=256,
    )

    var out_host = ctx.enqueue_create_host_buffer[DType.float32](num_rows * dim)
    var rms_host = ctx.enqueue_create_host_buffer[DType.float32](num_rows)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.enqueue_copy(dst_buf=rms_host, src_buf=rms_dev)
    ctx.synchronize()

    var pass_ = True
    if abs(rms_host[0] - 2.7386) > 0.01:
        print("  FAIL: rms[0]=", rms_host[0], "expected ~2.7386")
        pass_ = False
    if abs(rms_host[1] - 2.5) > 0.01:
        print("  FAIL: rms[1]=", rms_host[1], "expected ~2.5")
        pass_ = False
    if abs(out_host[0] - 1.0 / 2.7386) > 0.01:
        print("  FAIL: out[0,0]=", out_host[0], "expected ~0.3651")
        pass_ = False
    if abs(out_host[7] - 5.0 / 2.5) > 0.01:
        print("  FAIL: out[1,3]=", out_host[7], "expected ~2.0")
        pass_ = False
    if pass_:
        print("  PASS (rms=[", rms_host[0], ",", rms_host[1], "])")


def test_softmax(ctx: DeviceContext) raises:
    """Test row-wise softmax."""
    print("--- Test: Softmax Forward ---")
    comptime cols = 4
    comptime num_rows = 2

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](num_rows * cols)
    ctx.synchronize()
    in_host[0] = 1.0; in_host[1] = 2.0; in_host[2] = 3.0; in_host[3] = 4.0
    in_host[4] = 0.0; in_host[5] = 0.0; in_host[6] = 0.0; in_host[7] = 0.0

    var in_dev = ctx.enqueue_create_buffer[DType.float32](num_rows * cols)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](num_rows * cols)
    ctx.enqueue_copy(dst_buf=in_dev, src_buf=in_host)

    ctx.enqueue_function[softmax_fwd, softmax_fwd](
        out_dev, in_dev, num_rows, cols,
        grid_dim=num_rows, block_dim=256,
    )

    var out_host = ctx.enqueue_create_host_buffer[DType.float32](num_rows * cols)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(cols):
        if abs(out_host[cols + i] - 0.25) > 1e-5:
            print("  FAIL: softmax([0,0,0,0])[", i, "]=", out_host[cols + i], "expected 0.25")
            pass_ = False
    var row_sum: Float32 = 0.0
    for i in range(cols):
        row_sum += out_host[i]
    if abs(row_sum - 1.0) > 1e-5:
        print("  FAIL: row 0 sum =", row_sum, "expected 1.0")
        pass_ = False
    for i in range(cols - 1):
        if out_host[i] >= out_host[i + 1]:
            print("  FAIL: softmax not monotonic at", i)
            pass_ = False
    if pass_:
        print("  PASS (row0=[", out_host[0], out_host[1], out_host[2], out_host[3], "] row1=uniform)")


def test_cross_entropy(ctx: DeviceContext) raises:
    """Test cross-entropy loss."""
    print("--- Test: Cross-Entropy Forward ---")
    comptime vocab = 4
    comptime num_tokens = 2

    var logits_host = ctx.enqueue_create_host_buffer[DType.float32](num_tokens * vocab)
    var targets_host = ctx.enqueue_create_host_buffer[DType.int64](num_tokens)
    ctx.synchronize()
    logits_host[0] = 1.0; logits_host[1] = 2.0; logits_host[2] = 3.0; logits_host[3] = 4.0
    targets_host[0] = 3
    logits_host[4] = 4.0; logits_host[5] = 3.0; logits_host[6] = 2.0; logits_host[7] = 1.0
    targets_host[1] = 3

    var logits_dev = ctx.enqueue_create_buffer[DType.float32](num_tokens * vocab)
    var targets_dev = ctx.enqueue_create_buffer[DType.int64](num_tokens)
    var loss_dev = ctx.enqueue_create_buffer[DType.float32](num_tokens)
    ctx.enqueue_copy(dst_buf=logits_dev, src_buf=logits_host)
    ctx.enqueue_copy(dst_buf=targets_dev, src_buf=targets_host)

    ctx.enqueue_function[cross_entropy_fwd, cross_entropy_fwd](
        loss_dev, logits_dev, targets_dev, num_tokens, vocab,
        grid_dim=num_tokens, block_dim=256,
    )

    var loss_host = ctx.enqueue_create_host_buffer[DType.float32](num_tokens)
    ctx.enqueue_copy(dst_buf=loss_host, src_buf=loss_dev)
    ctx.synchronize()

    var pass_ = True
    if abs(loss_host[0] - 0.4402) > 0.01:
        print("  FAIL: loss[0]=", loss_host[0], "expected ~0.4402")
        pass_ = False
    if abs(loss_host[1] - 3.4402) > 0.01:
        print("  FAIL: loss[1]=", loss_host[1], "expected ~3.4402")
        pass_ = False
    if loss_host[0] >= loss_host[1]:
        print("  FAIL: loss[0] should be < loss[1]")
        pass_ = False
    if pass_:
        print("  PASS (loss=[", loss_host[0], ",", loss_host[1], "])")


def main() raises:
    print("=== Phase 1: Kernel Tests ===")
    print()

    comptime if not has_accelerator():
        print("ERROR: No GPU found")
        return

    var ctx = DeviceContext()

    test_embedding(ctx)
    test_relu_squared(ctx)
    test_softcap(ctx)
    test_rmsnorm(ctx)
    test_softmax(ctx)
    test_cross_entropy(ctx)

    print()
    print("=== All kernel tests complete ===")
