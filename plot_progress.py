"""Generate a progress chart from results.tsv.

Usage:
    pixi run python plot_progress.py

Reads results.tsv (tab-separated: commit, val_bpb, status, description)
and produces progress.png showing experiment progress over time.
"""

import sys
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pixi add matplotlib")
    sys.exit(1)


def main():
    tsv_path = os.path.join(os.path.dirname(__file__), "results.tsv")
    if not os.path.exists(tsv_path):
        print(f"No results.tsv found at {tsv_path}")
        print("Run some experiments first, then try again.")
        sys.exit(1)

    # Parse TSV
    experiments = []
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            commit, val_bpb, status, desc = parts[0], parts[1], parts[2], parts[3]
            try:
                bpb = float(val_bpb)
            except ValueError:
                bpb = 0.0
            experiments.append({
                "commit": commit,
                "val_bpb": bpb,
                "status": status.strip(),
                "description": desc.strip(),
            })

    if not experiments:
        print("No experiments found in results.tsv")
        sys.exit(1)

    # Separate by status
    kept = [(i, e) for i, e in enumerate(experiments) if e["status"] == "keep"]
    discarded = [(i, e) for i, e in enumerate(experiments) if e["status"] == "discard"]
    crashed = [(i, e) for i, e in enumerate(experiments) if e["status"] == "crash"]

    # Running minimum (frontier)
    running_min = []
    best = float("inf")
    for e in experiments:
        if e["status"] != "crash" and e["val_bpb"] > 0:
            best = min(best, e["val_bpb"])
        running_min.append(best if best < float("inf") else None)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Discarded (faint)
    if discarded:
        ax.scatter(
            [i for i, _ in discarded],
            [e["val_bpb"] for _, e in discarded],
            c="#cccccc", s=30, zorder=2, label=f"Discarded ({len(discarded)})",
        )

    # Crashed (red x)
    if crashed:
        ax.scatter(
            [i for i, _ in crashed],
            [0 for _ in crashed],
            c="red", marker="x", s=40, zorder=3, label=f"Crashed ({len(crashed)})",
        )

    # Kept (green)
    if kept:
        ax.scatter(
            [i for i, _ in kept],
            [e["val_bpb"] for _, e in kept],
            c="#2ecc71", s=60, zorder=4, label=f"Kept ({len(kept)})",
            edgecolors="black", linewidths=0.5,
        )

    # Running minimum frontier
    valid_frontier = [(i, v) for i, v in enumerate(running_min) if v is not None]
    if valid_frontier:
        ax.plot(
            [i for i, _ in valid_frontier],
            [v for _, v in valid_frontier],
            c="#e74c3c", linewidth=2, zorder=5, label="Best so far",
        )

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("val_bpb (lower is better)", fontsize=12)
    ax.set_title(
        f"Autoresearch Progress: {len(experiments)} experiments, "
        f"{len(kept)} kept improvements",
        fontsize=14,
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate best result
    if kept:
        best_exp = min(kept, key=lambda x: x[1]["val_bpb"])
        ax.annotate(
            f'{best_exp[1]["val_bpb"]:.4f}\n{best_exp[1]["description"][:30]}',
            xy=(best_exp[0], best_exp[1]["val_bpb"]),
            xytext=(10, -20), textcoords="offset points",
            fontsize=8, color="#2ecc71",
            arrowprops=dict(arrowstyle="->", color="#2ecc71"),
        )

    out_path = os.path.join(os.path.dirname(__file__), "progress.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    print(f"  {len(experiments)} total, {len(kept)} kept, "
          f"{len(discarded)} discarded, {len(crashed)} crashed")
    if valid_frontier:
        print(f"  Best val_bpb: {valid_frontier[-1][1]:.6f}")


if __name__ == "__main__":
    main()
