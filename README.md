# autoresearch

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

This is a **Mojo rewrite** of the original PyTorch training code, targeting cross-platform GPU support via Metal (Apple Silicon), CUDA (NVIDIA), and ROCm (AMD) through Mojo's unified `DeviceContext` abstraction. No autograd — all forward and backward passes are implemented as hand-written GPU kernels.

## How it works

The repo has a few key files:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.mojo`** — main training entry point. Training loop with AdamW + Muon optimizers, LR scheduling, time-budget enforcement, BPB evaluation.
- **`model.mojo`** — GPT model: weight allocation, initialization, forward pass, backward pass.
- **`kernels/ops.mojo`** — all GPU kernels: flash attention, tiled matmul (NT/NN/TN), RMSNorm, RoPE, embedding, cross-entropy, elementwise ops, and their backwards.
- **`optimizer.mojo`** — AdamW and Muon (Nesterov + Polar Express orthogonalization) optimizer kernels.
- **`config.mojo`** — model architecture config (GPTConfig) and training hyperparameters.
- **`program.md`** — agent instructions for autonomous experimentation.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup), regardless of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better.

## Quick start

**Requirements:** Apple Silicon Mac, Linux with NVIDIA/AMD GPU, [pixi](https://pixi.sh).

```bash
# 1. Install pixi (if you don't already have it)
curl -fsSL https://pixi.sh/install.sh | sh

# 2. Install dependencies (Mojo + Python packages)
pixi install

# 3. Download data and train tokenizer (one-time, ~2 min)
pixi run prepare

# 4. Run training (~5 min default, override with TRAIN_TIME_BUDGET)
pixi run train

# 5. Or build and run the binary directly
pixi run mojo build -o train train.mojo
TRAIN_TIME_BUDGET=300 pixi run ./train
```

### Overnight training

```bash
# 8 hours
TRAIN_TIME_BUDGET=28800 nohup pixi run ./train > train_overnight.log 2>&1 &

# Check progress
tail -5 train_overnight.log

# Stop early
pkill -f "./train"
```

## Architecture

The model is a pre-norm GPT with:
- RMSNorm (pre-norm before attention and MLP)
- Rotary position embeddings (RoPE)
- Flash Attention 2 (online softmax, O(T) memory)
- Squared ReLU activation (relu(x)^2)
- Sliding window attention (SSSL pattern)
- Value embeddings with gated residual (ResFormer)
- Learned per-layer residual scaling + skip connections
- Logit soft-capping (15 * tanh(x/15))

Optimizer: AdamW for embeddings/scalars, Muon (Polar Express orthogonalization) for matrix parameters.

## Project structure

```
prepare.py          — constants, data prep + runtime utilities (do not modify)
dataloader_cpu.py   — CPU-only dataloader wrapper for non-CUDA platforms
train.mojo          — training loop, LR schedule, BPB evaluation
model.mojo          — model weights, forward pass, backward pass
kernels/ops.mojo    — all GPU kernels (flash attention, tiled matmul, etc.)
optimizer.mojo      — AdamW + Muon optimizer kernels
config.mojo         — GPTConfig, hyperparameters
program.md          — agent instructions for autonomous experimentation
pixi.toml           — Mojo + Python dependencies
```

## Platform support

This Mojo rewrite runs on any platform with Mojo GPU support:
- **Apple Silicon** (Metal) — tested on M4 Max
- **NVIDIA** (CUDA) — should work via Mojo's DeviceContext
- **AMD** (ROCm) — should work via Mojo's DeviceContext

The original PyTorch version required NVIDIA CUDA + Flash Attention 3. This rewrite replaces all of that with hand-written Mojo GPU kernels that compile to Metal, CUDA, or ROCm.

### macOS notes

On macOS you may need to install the Metal Toolchain:
```bash
xcodebuild -downloadComponent MetalToolchain
```

## License

MIT
