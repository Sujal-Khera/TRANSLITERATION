"""
Training Visualization
=======================
Generates publication-quality plots from training history JSON files.

Generates:
    1. Loss Curves (Train vs Validation)
    2. Perplexity Curves
    3. Teacher Forcing Schedule
    4. Prediction Evolution (how outputs improve across epochs)
    5. Combined Dashboard (all in one figure)

Usage:
    python scripts/visualize_training.py stage1
    python scripts/visualize_training.py stage2
    python scripts/visualize_training.py stage3
    python scripts/visualize_training.py all
"""

import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

# ==========================================
# STYLE CONFIGURATION (3B1B-inspired dark theme)
# ==========================================
DARK_BG = "#1b1b2f"
CARD_BG = "#1a1a2e"
GRID_COLOR = "#2a2a3e"
TEXT_COLOR = "#e8e8f0"
DIM_TEXT = "#8888a0"
BLUE = "#6c63ff"
TEAL = "#00d2ff"
ORANGE = "#ff6b35"
GREEN = "#4ade80"
PINK = "#f472b6"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "xtick.color": DIM_TEXT,
    "ytick.color": DIM_TEXT,
    "text.color": TEXT_COLOR,
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Nirmala UI", "Mangal", "Arial Unicode MS", "DejaVu Sans"],
    "legend.facecolor": CARD_BG,
    "legend.edgecolor": GRID_COLOR,
    "legend.fontsize": 10,
})


def load_history(stage_name):
    """Load training history JSON for a given stage."""
    path = os.path.join(DATA_DIR, f"{stage_name}_history.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        print(f"Run training first: python scripts/train_{stage_name}.py")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curves(history, stage_name, ax=None):
    """Plot train vs validation loss curves."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=BLUE, linewidth=2.5, label="Train Loss", marker="o", markersize=4)
    ax.plot(epochs, history["val_loss"], color=ORANGE, linewidth=2.5, label="Validation Loss", marker="s", markersize=4)

    # Mark best epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val = min(history["val_loss"])
    ax.axvline(x=best_epoch, color=GREEN, linestyle="--", alpha=0.6, linewidth=1)
    ax.annotate(f"Best: {best_val:.3f}\n(Epoch {best_epoch})",
                xy=(best_epoch, best_val), xytext=(best_epoch + 1, best_val + 0.1),
                color=GREEN, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"{stage_name.upper()} — Loss Curves", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.legend()

    if standalone:
        save_path = os.path.join(DATA_DIR, f"{stage_name}_loss_curves.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)


def plot_perplexity(history, stage_name, ax=None):
    """Plot train vs validation perplexity."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_ppl"]) + 1)
    ax.plot(epochs, history["train_ppl"], color=TEAL, linewidth=2.5, label="Train PPL", marker="o", markersize=4)
    ax.plot(epochs, history["val_ppl"], color=PINK, linewidth=2.5, label="Validation PPL", marker="s", markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"{stage_name.upper()} — Perplexity", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.legend()

    # Log scale if values are large
    if max(history["train_ppl"]) > 100:
        ax.set_yscale("log")

    if standalone:
        save_path = os.path.join(DATA_DIR, f"{stage_name}_perplexity.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)


def plot_teacher_forcing(history, stage_name, ax=None):
    """Plot the teacher forcing ratio schedule."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    epochs = range(1, len(history["tf_ratio"]) + 1)
    ax.fill_between(epochs, history["tf_ratio"], alpha=0.3, color=BLUE)
    ax.plot(epochs, history["tf_ratio"], color=BLUE, linewidth=2.5, marker="o", markersize=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Teacher Forcing Ratio")
    ax.set_title(f"{stage_name.upper()} — Teacher Forcing Schedule", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylim(-0.05, 1.05)

    if standalone:
        save_path = os.path.join(DATA_DIR, f"{stage_name}_tf_schedule.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
        plt.close(fig)


def plot_prediction_evolution(history, stage_name):
    """Show how predictions evolve across training epochs (table-style image)."""
    predictions = history.get("predictions", [])
    if not predictions:
        print("  No prediction snapshots found. Skipping prediction evolution plot.")
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(predictions) * 1.2)))
    ax.axis("off")

    # Build table data
    sample_words = [s["input"] for s in predictions[0]["samples"]]
    col_labels = [f"Epoch {p['epoch']}" for p in predictions]
    header = ["Input"] + col_labels

    table_data = []
    for i, word in enumerate(sample_words):
        row = [word]
        for p in predictions:
            row.append(p["samples"][i]["prediction"])
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=header,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#2a2a4e")
            cell.set_text_props(color=TEAL, fontweight="bold")
        else:
            cell.set_facecolor(CARD_BG)
            cell.set_text_props(color=GREEN)

    ax.set_title(f"{stage_name.upper()} — Prediction Evolution",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=20)

    save_path = os.path.join(DATA_DIR, f"{stage_name}_predictions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_dashboard(history, stage_name):
    """Generate a combined 2x2 dashboard with all key plots."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{stage_name.upper()} Training Dashboard",
                 fontsize=20, fontweight="bold", color=TEXT_COLOR, y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_loss_curves(history, stage_name, ax=ax1)
    plot_perplexity(history, stage_name, ax=ax2)
    plot_teacher_forcing(history, stage_name, ax=ax3)

    # Summary stats in ax4
    ax4.axis("off")
    total_epochs = len(history["train_loss"])
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = min(history["val_loss"])
    best_val_ppl = history["val_ppl"][best_epoch - 1]
    final_train_loss = history["train_loss"][-1]
    overfit_gap = history["val_loss"][-1] - history["train_loss"][-1]

    stats_text = (
        f"Training Summary\n"
        f"{'─' * 30}\n"
        f"Total Epochs:       {total_epochs}\n"
        f"Best Epoch:         {best_epoch}\n"
        f"Best Val Loss:      {best_val_loss:.4f}\n"
        f"Best Val PPL:       {best_val_ppl:.2f}\n"
        f"Final Train Loss:   {final_train_loss:.4f}\n"
        f"Overfit Gap:        {overfit_gap:.4f}\n"
        f"{'─' * 30}\n"
    )

    # Color the overfit gap
    overfit_color = GREEN if abs(overfit_gap) < 0.2 else ORANGE if abs(overfit_gap) < 0.5 else PINK

    ax4.text(0.5, 0.6, stats_text, transform=ax4.transAxes,fontsize=13, fontfamily="monospace",
             color=TEXT_COLOR, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.8", facecolor=CARD_BG, edgecolor=GRID_COLOR))

    ax4.text(0.5, 0.15, f"Overfit Risk: {'LOW' if abs(overfit_gap) < 0.2 else 'MODERATE' if abs(overfit_gap) < 0.5 else 'HIGH'}",
             transform=ax4.transAxes, fontsize=14, fontweight="bold",
             color=overfit_color, ha="center")

    save_path = os.path.join(DATA_DIR, f"{stage_name}_dashboard.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def visualize_stage(stage_name):
    """Generate all visualizations for a training stage."""
    print(f"\n{'='*50}")
    print(f"  Generating {stage_name.upper()} Visualizations")
    print(f"{'='*50}\n")

    history = load_history(stage_name)
    if history is None:
        return

    print(f"  Loaded {len(history['train_loss'])} epochs of training data.\n")

    print("  [1/5] Loss Curves...")
    plot_loss_curves(history, stage_name)

    print("  [2/5] Perplexity...")
    plot_perplexity(history, stage_name)

    print("  [3/5] Teacher Forcing Schedule...")
    plot_teacher_forcing(history, stage_name)

    print("  [4/5] Prediction Evolution...")
    plot_prediction_evolution(history, stage_name)

    print("  [5/5] Combined Dashboard...")
    plot_dashboard(history, stage_name)

    print(f"\n✓ All {stage_name} visualizations saved to: {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training history")
    parser.add_argument("stage", nargs="?", default="stage1",
                        choices=["stage1", "stage2", "stage3", "all"],
                        help="Which stage to visualize (default: stage1)")
    args = parser.parse_args()

    if args.stage == "all":
        for stage in ["stage1", "stage2", "stage3"]:
            visualize_stage(stage)
    else:
        visualize_stage(args.stage)


if __name__ == "__main__":
    main()
