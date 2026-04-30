# CQRCD Paper Results Collection Guide

This document lists the exact results needed for the CQRCD research paper and the project files that should be run to collect them.

The paper needs evidence for three claims:

```text
1. DSRM creates detectable query-specific concentration.
2. CQRCD detects poisoned memories better than baseline defenses.
3. Evading CQRCD forces the attacker to sacrifice attack success.
```

## Before Running Experiments

Prepare the ASB-derived CQRCD dataset:

```bash
python -m scripts.prepare_asb_data
```

This creates:

```text
data/asb_tasks.json
data/legitimate_kb.json
data/adversarial_bb.json
data/adversarial_wb.json
```

For CPU-only smoke testing:

```bash
python -B -m experiments.run_sanity_check
```

For full local RTX testing, first validate the environment:

```bash
python run_setup_check.py
python test_local_setup.py
```

## Required Figures

All paper figures should be saved at 400 DPI in:

```text
results/figures/
```

| Figure | Script | Output File | Purpose |
|---|---|---|---|
| Figure 1 | `experiments/run_sanity_check.py` | `fig1_concentration_dist.png` | C-score separation |
| Figure 2 | `experiments/run_roc_analysis.py` | `fig2_roc_comparison.png` | Detection ROC/AUC comparison |
| Figure 3 | `experiments/run_main_experiment.py` | `fig3_asr_bar.png` | Attack success under defenses |
| Figure 4 | `experiments/run_adaptive_attack.py` | `fig4_adaptive_tradeoff.png` | Adaptive attacker tradeoff |
| Figure 5 | `experiments/run_ablation.py` | `fig5_ablation_n.png` | Neighbor-count ablation |
| Figure 6 | `experiments/run_ablation.py` | `fig6_ablation_threshold.png` | Threshold ablation |

## Required Tables

All paper tables should be saved in:

```text
results/tables/
```

| Table | Script | Output File | Purpose |
|---|---|---|---|
| Concentration summary | `experiments/run_sanity_check.py` | `concentration_summary.csv` | Mean/std C-score by document type |
| Detection metrics | `experiments/run_roc_analysis.py` | `detection_metrics.csv` | AUC, FNR, FPR, F1 |
| ASR results | `experiments/run_main_experiment.py` | `asr_results.csv` | ASR_A, ASR_R, RR by defense |
| Adaptive tradeoff | `experiments/run_adaptive_attack.py` | `adaptive_tradeoff.csv` | C-target vs ASR/detection |
| Ablation results | `experiments/run_ablation.py` | `ablation_results.csv` | Neighbor, threshold, retriever, metric ablations |

## Experiment 1: Concentration Score Separation

**Goal:** Show that adversarial DSRM documents have higher concentration scores than legitimate documents.

Run:

```bash
python -m experiments.run_sanity_check
```

Collect:

| Result | Meaning |
|---|---|
| Legitimate C-score mean/std | Expected around `1.0-1.3` |
| Black-box DSRM C-score mean/std | Expected above legitimate |
| White-box DSRM C-score mean/std | Expected highest |
| Distribution plot | Shows separation between clean and poisoned docs |

Required outputs:

```text
results/figures/fig1_concentration_dist.png
results/tables/concentration_summary.csv
```

Paper claim supported:

```text
DSRM creates query-specific relevance concentration that is measurable.
```

## Experiment 2: Detection Performance

**Goal:** Compare CQRCD against baseline defenses.

Run:

```bash
python -m experiments.run_roc_analysis
```

Collect:

| Metric | Meaning |
|---|---|
| CQRCD ROC-AUC | Main detection quality metric |
| PPL baseline ROC-AUC | Perplexity baseline comparison |
| LLM-detect FNR/FPR | LLM-based baseline comparison |
| CQRCD FNR at `tau_C = 1.65` | Missed adversarial documents |
| CQRCD FPR at `tau_C = 1.65` | Legitimate documents wrongly flagged |
| CQRCD F1 at `tau_C = 1.65` | Balanced detection score |

Required outputs:

```text
results/figures/fig2_roc_comparison.png
results/tables/detection_metrics.csv
```

Target values from the implementation plan:

| Metric | Target |
|---|---:|
| CQRCD AUC | `> 0.75`, expected about `0.81` |
| PPL AUC | about `0.49` |
| CQRCD FPR | `< 20%` |

Paper claim supported:

```text
CQRCD detects DSRM-style poisoned memories better than PPL and LLM-based baselines.
```

## Experiment 3: Attack Success Rate Under Defense

**Goal:** Measure whether CQRCD reduces actual tool-selection attack success in the RAG agent.

Run:

```bash
python -m experiments.run_main_experiment
```

Compare these conditions:

```text
No Defense
PPL Re-rank
LLM-detect
CQRCD
```

Collect:

| Metric | Meaning |
|---|---|
| `ASR_A` | Attack tool selected across all tasks |
| `ASR_R` | Attack retrieved and selected |
| `RR` | Adversarial document retrieval rate |
| `detection_rate` | Adversarial docs filtered |
| `FPR` | Legitimate docs wrongly filtered |

Required outputs:

```text
results/figures/fig3_asr_bar.png
results/tables/asr_results.csv
```

Target value from the implementation plan:

```text
ASR_A should drop by at least 40%.
Example target: from about 43% to below 26%.
```

Paper claim supported:

```text
CQRCD reduces end-to-end attack success, not only standalone detection score.
```

## Experiment 4: Adaptive Attacker Tradeoff

**Goal:** Show that when an attacker tries to evade CQRCD by lowering concentration, attack success also drops.

Run:

```bash
python -m experiments.run_adaptive_attack
```

Test concentration targets:

```text
1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 2.8
```

Collect:

| Result | Meaning |
|---|---|
| `C_target` | Attacker's maximum concentration target |
| `ASR_A` | Attack success under that constraint |
| `detection_rate` | How often CQRCD detects the constrained attacks |
| `mean_concentration` | Actual achieved C-score |

Required outputs:

```text
results/figures/fig4_adaptive_tradeoff.png
results/tables/adaptive_tradeoff.csv
```

Paper claim supported:

```text
Evading CQRCD forces a tradeoff: lower concentration also weakens retrieval and attack success.
```

## Experiment 5: Ablation Study

**Goal:** Show which CQRCD components matter and justify the default settings.

Run:

```bash
python -m experiments.run_ablation
```

Collect:

| Ablation | Values | Metrics |
|---|---|---|
| Neighbor count `n` | `1, 3, 5, 10, 20` | AUC, FNR, FPR |
| Neighbor method | `T5`, `embedding perturbation`, `random` | AUC |
| Threshold `tau_C` | `1.3, 1.4, 1.5, 1.65, 1.8, 2.0` | FNR + FPR |
| Similarity metric | `cosine`, `inner product`, `L2` | AUC |
| Retriever backbone | `DPR`, `MiniLM`, `ReaLM` | AUC, ASR_A |

Required outputs:

```text
results/figures/fig5_ablation_n.png
results/figures/fig6_ablation_threshold.png
results/tables/ablation_results.csv
```

Paper claims supported:

```text
n = 5 is near-optimal.
tau_C = 1.65 balances FNR and FPR.
CQRCD works across retriever backbones.
```

## Final Checklist

Before writing the paper results section, confirm these files exist:

```text
results/figures/fig1_concentration_dist.png
results/figures/fig2_roc_comparison.png
results/figures/fig3_asr_bar.png
results/figures/fig4_adaptive_tradeoff.png
results/figures/fig5_ablation_n.png
results/figures/fig6_ablation_threshold.png

results/tables/concentration_summary.csv
results/tables/detection_metrics.csv
results/tables/asr_results.csv
results/tables/adaptive_tradeoff.csv
results/tables/ablation_results.csv
```

## Suggested Run Order

Use this order because each step builds confidence before the more expensive runs:

```bash
python -m scripts.prepare_asb_data
python -m experiments.run_sanity_check
python -m experiments.run_roc_analysis
python -m experiments.run_main_experiment
python -m experiments.run_adaptive_attack
python -m experiments.run_ablation
python -m visualization.plot_results
```

## Notes

- The full RTX path should use the environment from `CQRCD_Local_Setup_RTX4070.md`.
- The CPU/mock path can verify project wiring, but it is not enough for final paper numbers.
- All reported paper figures should be generated from saved CSV/JSON results, not manually edited.
- Keep threshold calibration separate from the final test set to avoid data leakage.
