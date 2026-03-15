# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.mojo` — the main training entry point you can modify (loop, batch size, time budget).
   - `model.mojo` — the model you modify. Forward pass, backward pass, weight init.
   - `kernels/ops.mojo` — GPU kernels. Architecture changes go here.
   - `optimizer.mojo` — optimizer kernels (AdamW + Muon).
   - `config.mojo` — hyperparameters and model config.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `pixi run prepare`.
5. **Build**: Run `pixi run mojo build -o train train.mojo` to verify the code compiles.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it as:

```bash
pixi run mojo build -o train train.mojo && pixi run ./train > run.log 2>&1
```

**What you CAN do:**
- Modify `train.mojo` — training loop, batch size, gradient accumulation, LR schedule.
- Modify `model.mojo` — model architecture, forward/backward pass, weight init.
- Modify `kernels/ops.mojo` — GPU kernels (attention, matmul, normalization, etc.).
- Modify `optimizer.mojo` — optimizer algorithms and hyperparameters.
- Modify `config.mojo` — model dimensions, hyperparameter constants.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants.
- Install new packages or change `pixi.toml` dependencies.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the kernel implementations. The only constraint is that the code compiles, runs without crashing, and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will build and run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          2.515000
training_seconds: 300.1
total_seconds:    460.0
total_tokens_M:   13.1
num_steps:        25
depth:            8
```

You can extract the key metric:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

The TSV has a header row and 4 columns:

```
commit	val_bpb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar15`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify the Mojo source files with an experimental idea.
3. Build: `pixi run mojo build -o train train.mojo` — if it fails, fix and retry.
4. git commit
5. Run: `pixi run ./train > run.log 2>&1`
6. Read results: `grep "^val_bpb:" run.log`
7. If grep is empty, the run crashed. Run `tail -n 50 run.log` to diagnose.
8. Record results in the tsv
9. If val_bpb improved (lower), keep the commit
10. If val_bpb is equal or worse, `git reset --hard HEAD~1`

**Timeout**: Each experiment should take ~5 minutes total (+ startup overhead). If a run exceeds 10 minutes, kill it and treat as failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous. The loop runs until manually interrupted.
