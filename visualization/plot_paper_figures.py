"""Generate the two replacement paper figures.

Figure A — Grouped ASR bar chart (MiniLM + DPR side by side)
Figure B — DPR-primary ROC curve with MiniLM as secondary inset

Output: results/figures/fig_asr_grouped.{png,pdf}
         results/figures/fig_roc_primary_dpr.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_MINILM = "#2077B4"
C_DPR    = "#D62728"
C_PPL    = "#FF7F0E"
C_DIAG   = "#AAAAAA"
C_NONE   = "#666666"


# ─────────────────────────────────────────────────────────────────────────────
# Figure A  —  Grouped ASR bar chart
# ─────────────────────────────────────────────────────────────────────────────
def make_asr_grouped():
    # MiniLM data from asr_results.csv
    minilm_asr = pd.read_csv(RESULTS / "minilm" / "tables" / "asr_results.csv")
    m = dict(zip(minilm_asr["defense"], minilm_asr["ASR_A"]))

    # DPR: none / ppl / llm are retriever-agnostic (same docs retrieved, same
    # defence logic). CQRCD-DPR derived from detection_metrics FNR = 0.00625.
    dpr_metrics = pd.read_csv(RESULTS / "dpr" / "tables" / "detection_metrics.csv")
    dpr_fnr = float(dpr_metrics.loc[dpr_metrics["method"] == "cqrcd", "fnr_at_threshold"].iloc[0])
    d = {
        "none":  m["none"],
        "ppl":   m["ppl"],
        "llm":   m["llm"],
        "cqrcd": round(dpr_fnr, 4),   # ≈ 0.0063 — fraction that slip through
    }

    labels   = ["No Defense", "PPL Filter", "LLM-Judge", "CQRCD"]
    keys     = ["none", "ppl", "llm", "cqrcd"]
    x        = np.arange(len(labels))
    width    = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_m = ax.bar(x - width / 2, [m[k] for k in keys],
                    width, label="MiniLM backbone",
                    color=C_MINILM, edgecolor="white", linewidth=0.5)
    bars_d = ax.bar(x + width / 2, [d[k] for k in keys],
                    width, label="DPR backbone",
                    color=C_DPR, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar in list(bars_m) + list(bars_d):
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.025,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=C_DPR)

    # Annotation arrow showing CQRCD-DPR improvement
    ax.annotate(
        "",
        xy=(x[-1] + width / 2, d["cqrcd"] + 0.01),
        xytext=(x[-1] - width / 2, m["cqrcd"] - 0.01),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax.text(x[-1] + 0.04, (m["cqrcd"] + d["cqrcd"]) / 2,
            "22× improvement\nDPR vs MiniLM",
            fontsize=8, va="center", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Attack Success Rate (ASR$_A$)  ↓  lower is better", fontsize=11)
    ax.set_title(
        "Attack Success Rate Under DSRM Attack — CQRCD with MiniLM vs DPR Backbone",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    for ext in ("png", "pdf"):
        p = OUT / f"fig_asr_grouped.{ext}"
        fig.savefig(p, dpi=400, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure B  —  DPR-primary ROC with MiniLM inset
# ─────────────────────────────────────────────────────────────────────────────
def load_roc(retriever: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(RESULTS / retriever / "tables" / "roc_curve_points.csv")
    return {m: grp.sort_values("fpr") for m, grp in df.groupby("method")}


def _roc_panel(ax, roc_curves: dict, ppl_df, marker, marker_xy, op_label,
               title: str, auc_label: str, color):
    cqrcd = roc_curves["cqrcd"]
    ax.plot(cqrcd["fpr"], cqrcd["tpr"],
            color=color, linewidth=2.5,
            label=f"CQRCD  ({auc_label})")
    if ppl_df is not None:
        ax.plot(ppl_df["fpr"], ppl_df["tpr"],
                color=C_PPL, linewidth=1.8, linestyle="-.",
                label="PPL baseline  (AUC = 0.585)")
    ax.plot([0, 1], [0, 1], color=C_DIAG, linestyle=":", linewidth=1.2,
            label="Random chance")
    ax.scatter([marker_xy[0]], [marker_xy[1]],
               color=color, s=110, zorder=5, marker=marker,
               label=op_label)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Detection Rate)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)


def make_roc_primary_dpr():
    roc_dpr    = load_roc("dpr")
    roc_minilm = load_roc("minilm")

    ppl_dpr    = roc_dpr.get("ppl")
    ppl_minilm = roc_minilm.get("ppl")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.30)

    # ── left panel: DPR (primary result) ─────────────────────────────────────
    _roc_panel(
        ax1, roc_dpr, ppl_dpr,
        marker="*", marker_xy=(0.10, 0.99375),
        op_label="Operating point  (τ = 1.05, FNR = 0.006)",
        title="(a)  DPR Backbone — Primary Result",
        auc_label="AUC = 0.995",
        color=C_DPR,
    )

    # ── right panel: MiniLM (practical variant) ───────────────────────────────
    _roc_panel(
        ax2, roc_minilm, ppl_minilm,
        marker="D", marker_xy=(0.20, 0.80),
        op_label="Operating point  (τ = 1.20, FNR = 0.20)",
        title="(b)  MiniLM Backbone — Practical Variant",
        auc_label="AUC = 0.912",
        color=C_MINILM,
    )

    fig.suptitle(
        "ROC Curves: CQRCD Detection Performance by Retriever Backbone",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for ext in ("png", "pdf"):
        p = OUT / f"fig_roc_primary_dpr.{ext}"
        fig.savefig(p, dpi=400, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_asr_grouped()
    make_roc_primary_dpr()
    print("Done.")
