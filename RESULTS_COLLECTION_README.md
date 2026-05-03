# CQRCD Paper Results Collection Guide

This document lists the exact results needed for the CQRCD research paper and the project files that should be run to collect them.

The paper needs evidence for three claims:

```text
1. DSRM creates detectable query-specific concentration.
2. CQRCD detects poisoned memories better than baseline defenses.
3. CQRCD remains strong on standard attacks, while adaptive conclusions must be scoped by retriever.
```

For submission-quality reporting, treat `results/dpr/` as the canonical paper artifact set. Use `results/minilm/` for development, practical comparison, and limitation analysis.

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
python -m experiments.run_sanity_check --smoke
python -m experiments.run_roc_analysis --smoke
python -m experiments.run_main_experiment --smoke
```

For full local RTX testing, first validate the environment:

```bash
python run_setup_check.py
python test_local_setup.py
```

## Required Figures

Canonical DPR paper figures should be saved at 400 DPI in:

```text
results/dpr/figures/
```

| Figure | Script | Output File | Purpose |
|---|---|---|---|
| Figure 1 | `experiments/run_sanity_check.py` | `fig1_concentration_dist.png` | DPR concentration separation |
| Figure 2 | `experiments/run_roc_analysis.py` | `fig2_roc_comparison.png` | DPR detection ROC/AUC comparison |
| Figure 3 | `experiments/run_main_experiment.py` | `fig3_asr_bar.png` | DPR attack success under defenses |
| Figure 4 | `experiments/run_adaptive_attack.py` | `fig4_adaptive_tradeoff.png` | MiniLM adaptive limitation analysis |
| Figure 5 | `experiments/run_ablation.py` | `fig5_ablation_n.png` | DPR neighbor-count ablation |
| Figure 6 | `experiments/run_ablation.py` | `fig6_ablation_threshold.png` | DPR threshold ablation |
| Figure 7 | `experiments/run_ablation.py` | `fig7_ablation_method.png` | Neighbor-method ablation |

## Required Tables

Canonical DPR paper tables should be saved in:

```text
results/dpr/tables/
```

| Table | Script | Output File | Purpose |
|---|---|---|---|
| Concentration summary | `experiments/run_sanity_check.py` | `concentration_summary.csv` | Mean/std C-score by document type |
| Detection metrics | `experiments/run_roc_analysis.py` | `detection_metrics.csv` | AUC, FNR, FPR, F1 |
| ASR results | `experiments/run_main_experiment.py` | `asr_results.csv` | ASR_A, ASR_R, RR by defense |
| Ablation results | `experiments/run_ablation.py` | `ablation_results.csv` | Neighbor, threshold, retriever, method ablations |
| MiniLM adaptive tradeoff | `experiments/run_adaptive_attack.py` | `adaptive_tradeoff.csv` | C-target vs ASR/detection under MiniLM |
| MiniLM adaptive diagnostics | `experiments/run_adaptive_attack.py` | `adaptive_tradeoff_diagnostics.csv` | Per-example target mismatch and candidate-pool evidence |

## Experiment 1: Concentration Score Separation

**Goal:** Show that adversarial DSRM documents have higher concentration scores than legitimate documents.

Run for DPR:

```bash
python -m experiments.run_sanity_check --output-dir results/dpr
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
results/dpr/figures/fig1_concentration_dist.png
results/dpr/tables/concentration_summary.csv
```

Paper claim supported:

```text
DSRM creates query-specific relevance concentration that is measurable.
```

## Experiment 2: Detection Performance

**Goal:** Compare CQRCD against baseline defenses.

Run for DPR:

```bash
python -m experiments.run_roc_analysis --retriever dpr --output-dir results/dpr
```

Collect:

| Metric | Meaning |
|---|---|
| CQRCD ROC-AUC | Main detection quality metric |
| PPL baseline ROC-AUC | Perplexity baseline comparison |
| LLM-detect FNR/FPR | LLM-based baseline comparison |
| CQRCD FNR at calibrated `tau_C` | Missed adversarial documents |
| CQRCD FPR at calibrated `tau_C` | Legitimate documents wrongly flagged |
| CQRCD F1 at calibrated `tau_C` | Balanced detection score |

Required outputs:

```text
results/dpr/figures/fig2_roc_comparison.png
results/dpr/tables/detection_metrics.csv
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

Run for DPR:

```bash
python -m experiments.run_main_experiment --retriever dpr --output-dir results/dpr
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
results/dpr/figures/fig3_asr_bar.png
results/dpr/tables/asr_results.csv
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

**Goal:** Measure adaptive behavior honestly and scope conclusions by retriever. Current MiniLM results show strong standard detection but meaningful adaptive evasion at low concentration targets.

Run for MiniLM limitation analysis:

```bash
python -m experiments.run_adaptive_attack --retriever minilm --output-dir results/minilm
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
| `mean_abs_target_error` | Aggregate target mismatch |
| `adaptive_tradeoff_diagnostics.csv` | Per-example ceiling/selection evidence |

Required outputs:

```text
results/minilm/figures/fig4_adaptive_tradeoff.png
results/minilm/tables/adaptive_tradeoff.csv
results/minilm/tables/adaptive_tradeoff_diagnostics.csv
```

Paper claim supported:

```text
MiniLM adaptive results must be reported as a limitation analysis, not as a generic proof that evasion always degrades attack success.
```

## Experiment 5: Ablation Study

**Goal:** Show which CQRCD components matter and justify the default settings.

Run for DPR:

```bash
python -m experiments.run_ablation --retriever dpr --output-dir results/dpr
```

Collect:

| Ablation | Values | Metrics |
|---|---|---|
| Neighbor count `n` | `1, 3, 5, 10, 20` | ROC-AUC, threshold metrics |
| Neighbor method | `mixed`, `thematic_only`, `synonym_only`, `t5_mixed` | ROC-AUC |
| Threshold `tau_C` | `1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 2.0` | FNR + FPR |
| Retriever backbone | `DPR`, `MiniLM` | ROC-AUC, threshold metrics |

Required outputs:

```text
results/dpr/figures/fig5_ablation_n.png
results/dpr/figures/fig6_ablation_threshold.png
results/dpr/figures/fig7_ablation_method.png
results/dpr/tables/ablation_results.csv
```

Paper claims supported:

```text
n = 5 is near-optimal.
thematic neighbors can be compared directly against synonym-only neighbors under both DPR and MiniLM.
CQRCD behavior depends on retriever geometry and threshold calibration.
```

## Final Checklist

Before writing the paper results section, confirm these files exist:

```text
results/dpr/figures/fig1_concentration_dist.png
results/dpr/figures/fig2_roc_comparison.png
results/dpr/figures/fig3_asr_bar.png
results/dpr/figures/fig5_ablation_n.png
results/dpr/figures/fig6_ablation_threshold.png
results/dpr/figures/fig7_ablation_method.png

results/dpr/tables/concentration_summary.csv
results/dpr/tables/detection_metrics.csv
results/dpr/tables/asr_results.csv
results/dpr/tables/ablation_results.csv

results/minilm/figures/fig4_adaptive_tradeoff.png
results/minilm/tables/adaptive_tradeoff.csv
results/minilm/tables/adaptive_tradeoff_diagnostics.csv
```

## Suggested Run Order

Use this order because each step builds confidence before the more expensive runs:

```bash
python -m scripts.prepare_asb_data
python -m experiments.run_sanity_check --output-dir results/dpr
python -m experiments.run_roc_analysis --retriever dpr --output-dir results/dpr
python -m experiments.run_main_experiment --retriever dpr --output-dir results/dpr
python -m experiments.run_ablation --retriever dpr --output-dir results/dpr
python -m visualization.plot_results --input-dir results/dpr

python -m experiments.run_adaptive_attack --retriever minilm --output-dir results/minilm
python -m visualization.plot_results --input-dir results/minilm
```

## Notes

- The full RTX path should use the environment from `CQRCD_Local_Setup_RTX4070.md`.
- The CPU/mock path can verify project wiring, but it is not enough for final paper numbers.
- DPR is the canonical paper retriever; MiniLM is the practical secondary retriever used for speed and limitation analysis.
- Use `changes.md` as the repo-level change/debug history before re-tracing old fixes by hand.
- All reported paper figures should be generated from saved CSV/JSON results, not manually edited.
- Keep threshold calibration separate from the final test set to avoid data leakage.
- A neighbor-aware adaptive attacker against the thematic template family is still untested and should be stated as an open limitation unless explicitly evaluated.
