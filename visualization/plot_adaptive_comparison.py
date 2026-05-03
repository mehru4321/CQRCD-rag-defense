"""Figure: MiniLM vs DPR adaptive attacker tradeoff — two-panel comparison.

Saves to results/figures/fig_adaptive_backbone_comparison.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


RESULTS_ROOT = Path("results")
OUT_DIR = RESULTS_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load(retriever: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_ROOT / retriever / "tables" / filename)


def draw_panel(ax, df: pd.DataFrame, threshold: float, title: str, show_ylabel: bool):
    blue = "#2077B4"
    orange = "#FF7F0E"

    ax.plot(df["c_target"], df["ASR_A"],
            marker="o", color=blue, linewidth=2, markersize=6,
            label="Attack Success Rate (ASR$_A$)")
    ax.plot(df["c_target"], df["detection_rate"],
            marker="s", color=orange, linewidth=2, markersize=6,
            label="Detection Rate")

    ax.axvline(x=threshold, color="crimson", linestyle="--", linewidth=1.5, alpha=0.85,
               label=f"Threshold τ = {threshold}")

    ax.fill_between(df["c_target"], df["detection_rate"], 1.0,
                    alpha=0.07, color=orange, label="_nolegend_")
    ax.fill_between(df["c_target"], 0, df["ASR_A"],
                    alpha=0.07, color=blue, label="_nolegend_")

    ax.set_xlim(df["c_target"].min() - 0.05, df["c_target"].max() + 0.05)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("Target Concentration ($c_{target}$)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(fontsize=8.5, loc="center right")

    # Annotate converged values
    last = df.iloc[-1]
    ax.annotate(f"ASR ≈ {last['ASR_A']:.3f}",
                xy=(last["c_target"], last["ASR_A"]),
                xytext=(-48, 10), textcoords="offset points",
                fontsize=8, color=blue,
                arrowprops=dict(arrowstyle="->", color=blue, lw=0.8))
    ax.annotate(f"Det ≈ {last['detection_rate']:.3f}",
                xy=(last["c_target"], last["detection_rate"]),
                xytext=(-48, -18), textcoords="offset points",
                fontsize=8, color=orange,
                arrowprops=dict(arrowstyle="->", color=orange, lw=0.8))


def main():
    minilm = load("minilm", "adaptive_tradeoff.csv")
    dpr    = load("dpr",    "adaptive_tradeoff.csv")

    fig = plt.figure(figsize=(13, 4.8))
    gs  = gridspec.GridSpec(1, 2, wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    draw_panel(ax1, minilm, threshold=1.20,
               title="MiniLM  (AUC = 0.912, τ = 1.20)\nSharp transition — evasion possible below threshold",
               show_ylabel=True)

    draw_panel(ax2, dpr, threshold=1.05,
               title="DPR  (AUC = 0.995, τ = 1.05)\nNear-total detection across all attacker targets",
               show_ylabel=False)

    fig.suptitle(
        "Adaptive Attacker Tradeoff: MiniLM vs DPR Backbone",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ext in ("png", "pdf"):
        path = OUT_DIR / f"fig_adaptive_backbone_comparison.{ext}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
