"""
Autoresearch pretraining script — tinygrad backend (Metal GPU).
Mirrors train.py architecture: pre-norm GPT with RoPE, sliding window attention,
value embeddings with gated residual, learned residual scaling, squared ReLU, logit soft-capping.

Usage: python train_tinygrad.py
"""

import os
os.environ["METAL"] = "1"

import sys
sys.path.insert(0, "/Users/tsv/Developer/tinygrad")

import math
import time
import numpy as np

from tinygrad import Tensor, dtypes, nn, Device
from tinygrad.nn.optim import AdamW, OptimizerGroup
from tinygrad.nn.state import get_state_dict, get_parameters

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, get_token_bytes
from dataloader_cpu import make_dataloader_cpu

print(f"tinygrad device: {Device.DEFAULT}")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEPTH = 4
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"
MLP_EXPANSION = 4

TOTAL_BATCH_SIZE = 4 * 2048 * 4  # 32K tokens per optimizer step
DEVICE_BATCH_SIZE = 4    # per-device batch size

EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
ADAM_BETAS = (0.8, 0.95)
WEIGHT_DECAY = 0.2

WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def rms_norm(x: Tensor) -> Tensor:
    return x * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt().cast(x.dtype)

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, T, n_head, head_dim)
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return y1.cat(y2, dim=3)


class CausalSelfAttention:
    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, n_layer):
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_q = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, n_kv_head, bias=False) if has_ve(layer_idx, n_layer) else None

    def __call__(self, x: Tensor, ve: Tensor | None, cos: Tensor, sin: Tensor, window_size: int) -> Tensor:
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual with gated mixing
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = (self.ve_gate(x[..., :self.ve_gate_channels])).sigmoid() * 2  # (B, T, n_kv_head)
            v = v + gate.unsqueeze(-1) * ve

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm
        q = rms_norm(q)
        k = rms_norm(k)

        # Attention: (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA expand if needed (reshape + expand pattern since tinygrad lacks repeat_interleave)
        if self.n_kv_head < self.n_head:
            repeats = self.n_head // self.n_kv_head
            # k: (B, n_kv_head, T, D) -> (B, n_kv_head, 1, T, D) -> (B, n_kv_head, repeats, T, D) -> (B, n_head, T, D)
            k = k.unsqueeze(2).expand(B, self.n_kv_head, repeats, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_head, repeats, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)

        # Scaled dot-product attention with causal mask and sliding window
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = q.matmul(k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Causal + sliding window mask
        # Build mask: causal + window constraint
        mask = Tensor.ones(T, T, requires_grad=False).tril()
        if window_size < T:
            # Zero out positions more than window_size away
            window_mask = Tensor.ones(T, T, requires_grad=False).triu(-window_size + 1)
            mask = mask * window_mask
        mask = mask.where(0, float("-inf"))

        attn = attn + mask
        attn = attn.cast(dtypes.float32).softmax(-1).cast(x.dtype)

        y = attn.matmul(v)  # (B, H, T, D)
        y = y.transpose(1, 2).reshape(B, T, -1)  # (B, T, C)
        return self.c_proj(y)


class MLP:
    def __init__(self, n_embd, expansion=4):
        self.c_fc = nn.Linear(n_embd, expansion * n_embd, bias=False)
        self.c_proj = nn.Linear(expansion * n_embd, n_embd, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = x.relu().square()  # squared ReLU
        return self.c_proj(x)


class Block:
    def __init__(self, n_embd, n_head, n_kv_head, layer_idx, n_layer, expansion=4):
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, layer_idx, n_layer)
        self.mlp = MLP(n_embd, expansion)

    def __call__(self, x: Tensor, ve: Tensor | None, cos: Tensor, sin: Tensor, window_size: int) -> Tensor:
        x = x + self.attn(rms_norm(x), ve, cos, sin, window_size)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT:
    def __init__(self, vocab_size, n_layer, n_embd, n_head, n_kv_head, seq_len, window_pattern="SSSL"):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.seq_len = seq_len
        self.head_dim = n_embd // n_head

        # Embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)

        # Transformer blocks
        self.blocks = [Block(n_embd, n_head, n_kv_head, i, n_layer) for i in range(n_layer)]

        # LM head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Per-layer residual scaling
        self.resid_lambdas = Tensor.ones(n_layer, requires_grad=True)
        self.x0_lambdas = Tensor.zeros(n_layer, requires_grad=True)

        # Value embeddings (alternating layers)
        kv_dim = n_kv_head * self.head_dim
        self.value_embeds = {}
        for i in range(n_layer):
            if has_ve(i, n_layer):
                self.value_embeds[str(i)] = nn.Embedding(vocab_size, kv_dim)

        # Sliding window pattern
        self.window_sizes = self._compute_window_sizes(window_pattern)

        # Precompute RoPE
        self.cos, self.sin = self._precompute_rotary(seq_len * 10, self.head_dim)

    def _compute_window_sizes(self, pattern):
        long_window = self.seq_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(self.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = long_window  # last layer always full
        return window_sizes

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        channel_range = np.arange(0, head_dim, 2, dtype=np.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        cos_np = np.cos(freqs).astype(np.float32)
        sin_np = np.sin(freqs).astype(np.float32)
        # Shape: (1, seq_len, 1, head_dim//2)
        cos = Tensor(cos_np, requires_grad=False).reshape(1, seq_len, 1, head_dim // 2).cast(dtypes.bfloat16)
        sin = Tensor(sin_np, requires_grad=False).reshape(1, seq_len, 1, head_dim // 2).cast(dtypes.bfloat16)
        return cos, sin

    def init_weights(self):
        """Custom weight initialization matching train.py."""
        n_embd = self.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding: normal(0, 1) then cast to bf16
        self.wte.weight = Tensor.randn(self.vocab_size, n_embd).cast(dtypes.bfloat16).requires_grad_()

        # LM head: normal(0, 0.001)
        self.lm_head.weight = (Tensor.randn(self.vocab_size, n_embd) * 0.001).requires_grad_()

        # Transformer blocks
        for block in self.blocks:
            attn = block.attn
            attn.c_q.weight = Tensor.uniform(attn.c_q.weight.shape[0], attn.c_q.weight.shape[1], low=-s, high=s).requires_grad_()
            attn.c_k.weight = Tensor.uniform(attn.c_k.weight.shape[0], attn.c_k.weight.shape[1], low=-s, high=s).requires_grad_()
            attn.c_v.weight = Tensor.uniform(attn.c_v.weight.shape[0], attn.c_v.weight.shape[1], low=-s, high=s).requires_grad_()
            attn.c_proj.weight = Tensor.zeros(attn.c_proj.weight.shape[0], attn.c_proj.weight.shape[1]).requires_grad_()
            block.mlp.c_fc.weight = Tensor.uniform(block.mlp.c_fc.weight.shape[0], block.mlp.c_fc.weight.shape[1], low=-s, high=s).requires_grad_()
            block.mlp.c_proj.weight = Tensor.zeros(block.mlp.c_proj.weight.shape[0], block.mlp.c_proj.weight.shape[1]).requires_grad_()
            # VE gate init to zero
            if attn.ve_gate is not None:
                attn.ve_gate.weight = Tensor.zeros(attn.ve_gate.weight.shape[0], attn.ve_gate.weight.shape[1]).requires_grad_()

        # Per-layer scalars
        self.resid_lambdas = Tensor.ones(self.n_layer).requires_grad_()
        self.x0_lambdas = Tensor.full((self.n_layer,), 0.1).requires_grad_()

        # Value embeddings
        for key, ve in self.value_embeds.items():
            ve.weight = Tensor.uniform(self.vocab_size, ve.weight.shape[1], low=-s, high=s).cast(dtypes.bfloat16).requires_grad_()

    def __call__(self, idx: Tensor, targets: Tensor | None = None, reduction: str = "mean") -> Tensor:
        B, T = idx.shape

        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        x = self.wte(idx)
        x = rms_norm(x)
        x0 = x

        for i, block in enumerate(self.blocks):
            rl = self.resid_lambdas[i]
            xl = self.x0_lambdas[i]
            x = rl * x + xl * x0

            ve_key = str(i)
            ve = self.value_embeds[ve_key](idx) if ve_key in self.value_embeds else None
            x = block(x, ve, cos, sin, self.window_sizes[i])

        x = rms_norm(x)

        # LM head + soft-capping
        logits = self.lm_head(x)
        logits = logits.float()
        logits = 15.0 * (logits / 15.0).tanh()

        if targets is not None:
            loss = logits.reshape(-1, logits.shape[-1]).cross_entropy(targets.reshape(-1), reduction=reduction)
            return loss
        return logits


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

Tensor.manual_seed(42)
t_start = time.time()

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Build model config
base_dim = DEPTH * ASPECT_RATIO
model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
num_heads = model_dim // HEAD_DIM
n_kv_head = num_heads

print(f"Model: depth={DEPTH}, dim={model_dim}, heads={num_heads}, kv_heads={n_kv_head}, head_dim={HEAD_DIM}")

model = GPT(
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_embd=model_dim,
    n_head=num_heads,
    n_kv_head=n_kv_head,
    seq_len=MAX_SEQ_LEN,
    window_pattern=WINDOW_PATTERN,
)
model.init_weights()

# Count parameters
all_params = get_parameters(model)
num_params = sum(p.numel() for p in all_params)
print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")

# Batch size and gradient accumulation
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, f"TOTAL_BATCH_SIZE ({TOTAL_BATCH_SIZE}) must be divisible by tokens_per_fwdbwd ({tokens_per_fwdbwd})"
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Optimizer: AdamW with per-group learning rates
# ---------------------------------------------------------------------------

dmodel_lr_scale = (model_dim / 768) ** -0.5
print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

# Categorize parameters
embedding_params = [model.wte.weight]
value_embed_params = [ve.weight for ve in model.value_embeds.values()]
lm_head_params = [model.lm_head.weight]
resid_params = [model.resid_lambdas]
x0_params = [model.x0_lambdas]

# Collect all matrix params from blocks
matrix_params = []
for block in model.blocks:
    matrix_params.extend([
        block.attn.c_q.weight, block.attn.c_k.weight, block.attn.c_v.weight, block.attn.c_proj.weight,
        block.mlp.c_fc.weight, block.mlp.c_proj.weight,
    ])
    if block.attn.ve_gate is not None:
        matrix_params.append(block.attn.ve_gate.weight)

# Create optimizer groups
opt_embedding = AdamW(embedding_params, lr=EMBEDDING_LR * dmodel_lr_scale, b1=ADAM_BETAS[0], b2=ADAM_BETAS[1], eps=1e-10, weight_decay=0.0)
opt_ve = AdamW(value_embed_params, lr=EMBEDDING_LR * dmodel_lr_scale, b1=ADAM_BETAS[0], b2=ADAM_BETAS[1], eps=1e-10, weight_decay=0.0) if value_embed_params else None
opt_lm_head = AdamW(lm_head_params, lr=UNEMBEDDING_LR * dmodel_lr_scale, b1=ADAM_BETAS[0], b2=ADAM_BETAS[1], eps=1e-10, weight_decay=0.0)
opt_resid = AdamW(resid_params, lr=SCALAR_LR * 0.01, b1=ADAM_BETAS[0], b2=ADAM_BETAS[1], eps=1e-10, weight_decay=0.0)
opt_x0 = AdamW(x0_params, lr=SCALAR_LR, b1=0.96, b2=ADAM_BETAS[1], eps=1e-10, weight_decay=0.0)
opt_matrix = AdamW(matrix_params, lr=MATRIX_LR * dmodel_lr_scale, b1=ADAM_BETAS[0], b2=ADAM_BETAS[1], eps=1e-10, weight_decay=WEIGHT_DECAY)

optimizers = [opt_embedding, opt_lm_head, opt_resid, opt_x0, opt_matrix]
if opt_ve is not None:
    optimizers.append(opt_ve)
optimizer = OptimizerGroup(*optimizers)

# Store initial LRs for scheduling
initial_lrs = [o.lr.numpy().item() for o in optimizers]

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

T = MAX_SEQ_LEN
B = DEVICE_BATCH_SIZE

train_loader = make_dataloader_cpu(tokenizer, B, T, "train")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Sequence length: {T}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

Tensor.training = True
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

print("Starting training...")

while True:
    t0 = time.time()

    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        x_np, y_np, epoch = next(train_loader)
        x_t = Tensor(x_np.numpy().copy(), requires_grad=False).cast(dtypes.int32)
        y_t = Tensor(y_np.numpy().copy(), requires_grad=False).cast(dtypes.int32)

        loss = model(x_t, y_t)
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()
        # Realize loss + all grads to keep compute graph bounded during accumulation
        grads_to_realize = [p.grad for p in all_params if p.grad is not None]
        scaled_loss.realize(*grads_to_realize)

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0.0
    lrm = get_lr_multiplier(progress)
    for opt, init_lr in zip(optimizers, initial_lrs):
        opt.lr.assign(Tensor([init_lr * lrm], requires_grad=False, dtype=opt.lr.dtype))

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    train_loss_f = loss.numpy().item()

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL: loss exploded")
        sys.exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# ---------------------------------------------------------------------------
# Evaluation: BPB on validation set
# ---------------------------------------------------------------------------

print("Evaluating validation BPB...")

Tensor.training = False

# Load token byte lengths (from torch tensor -> numpy -> tinygrad)
import torch as _torch
token_bytes_torch = get_token_bytes(device="cpu")
token_bytes_np = token_bytes_torch.numpy().astype(np.int32)

EVAL_TOKENS = 40 * 524288
eval_steps = EVAL_TOKENS // (B * T)
val_loader = make_dataloader_cpu(tokenizer, B, T, "val")

total_nats = 0.0
total_bytes = 0

for eval_step in range(eval_steps):
    x_np, y_np, _ = next(val_loader)
    x_np_copy = x_np.numpy().copy()
    y_np_copy = y_np.numpy().copy()
    x_t = Tensor(x_np_copy, requires_grad=False).cast(dtypes.int32)
    y_t = Tensor(y_np_copy, requires_grad=False).cast(dtypes.int32)

    loss_flat = model(x_t, y_t, reduction="none").reshape(-1)
    y_flat = y_np_copy.reshape(-1)

    # Get per-token losses
    loss_np = loss_flat.numpy()

    # Get byte lengths for target tokens
    nbytes = token_bytes_np[y_flat]
    mask = nbytes > 0

    total_nats += (loss_np * mask).sum()
    total_bytes += nbytes.sum()

    if (eval_step + 1) % 10 == 0:
        print(f"\r  eval step {eval_step+1}/{eval_steps}", end="", flush=True)

print()

val_bpb = total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()

print("---")
print(f"val_bpb:          {val_bpb:.4f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"depth:            {DEPTH}")
