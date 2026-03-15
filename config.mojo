# Model architecture and training hyperparameters.
#
# All constants that control model shape, optimization, and training behavior
# live here. This mirrors the hyperparameter section of the original train.py.


# ── Model architecture ───────────────────────────────────────────────────────

@fieldwise_init
struct GPTConfig(ImplicitlyCopyable, Copyable, Movable):
    """GPT model architecture configuration.

    Fields are derived from depth via `default()`, following the scaling rules:
      model_dim = depth * 64, rounded up to nearest multiple of HEAD_DIM (128).
    """

    var sequence_len: Int
    var vocab_size: Int
    var n_layer: Int
    var n_head: Int
    var n_kv_head: Int
    var n_embd: Int
    var head_dim: Int
    var kv_dim: Int

    @staticmethod
    fn default(vocab_size: Int, depth: Int) -> Self:
        """Build config from vocab size and depth using standard scaling rules."""
        comptime ASPECT_RATIO = 64
        comptime HEAD_DIM = 128
        var base_dim = depth * ASPECT_RATIO
        var model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
        var num_heads = model_dim // HEAD_DIM
        return Self(
            sequence_len=2048,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
            head_dim=model_dim // num_heads,
            kv_dim=num_heads * (model_dim // num_heads),
        )

    fn has_ve(self, layer_idx: Int) -> Bool:
        """Whether this layer has a value embedding (alternating, last always on)."""
        return layer_idx % 2 == (self.n_layer - 1) % 2

    fn window_size(self, layer_idx: Int) -> Int:
        """Attention window size: pattern SSSL (S=half, L=full context)."""
        if layer_idx == self.n_layer - 1:
            return self.sequence_len
        if layer_idx % 4 < 3:
            return self.sequence_len // 2
        return self.sequence_len


# ── Training hyperparameters ─────────────────────────────────────────────────

comptime MAX_SEQ_LEN: Int = 2048
comptime TIME_BUDGET: Int = 300     # seconds (5 minutes)
comptime DEPTH: Int = 8
comptime MLP_EXPANSION: Int = 4
comptime VE_GATE_CHANNELS: Int = 32

# Effective batch size (tokens per optimizer step)
comptime TOTAL_BATCH_SIZE: Int = 1 << 19  # ~524K

# Learning rates (per parameter group)
comptime EMBEDDING_LR: Float64 = 0.6
comptime UNEMBEDDING_LR: Float64 = 0.004
comptime MATRIX_LR: Float64 = 0.04
comptime SCALAR_LR: Float64 = 0.5

# AdamW / Muon
comptime ADAM_BETA1: Float64 = 0.8
comptime ADAM_BETA2: Float64 = 0.95
comptime WEIGHT_DECAY: Float64 = 0.2

# LR schedule
comptime WARMUP_RATIO: Float64 = 0.0
comptime WARMDOWN_RATIO: Float64 = 0.5
comptime FINAL_LR_FRAC: Float64 = 0.0

# Architecture constants
comptime SOFTCAP: Float64 = 15.0
comptime ROPE_BASE: Float64 = 10000.0


# ── Polar Express coefficients (Muon optimizer, 5 iterations) ────────────────
# Precomputed power-series coefficients for the Cayley transform.

fn POLAR_COEFF_A(i: Int) -> Float64:
    if i == 0: return 8.156554524902461
    if i == 1: return 4.042929935166739
    if i == 2: return 3.8916678022926607
    if i == 3: return 3.285753657755655
    return 2.3465413258596377

fn POLAR_COEFF_B(i: Int) -> Float64:
    if i == 0: return -22.48329292557795
    if i == 1: return -2.808917465908714
    if i == 2: return -2.772484153217685
    if i == 3: return -2.3681294933425376
    return -1.7097828382687081

fn POLAR_COEFF_C(i: Int) -> Float64:
    if i == 0: return 15.878769915207462
    if i == 1: return 0.5000178451051316
    if i == 2: return 0.5060648178503393
    if i == 3: return 0.46449024233003106
    return 0.42323551169305323
