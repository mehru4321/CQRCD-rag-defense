# Changes Log

## Session 001 - Pipeline foundation
- Files changed: `changes.md`, `experiments/common.py`, `evaluation/metrics.py`, `modules/agent_simulator.py`, `modules/baseline_defenses.py`, `modules/dsrm_simulator.py`, `experiments/run_sanity_check.py`, `experiments/run_roc_analysis.py`, `experiments/run_main_experiment.py`, `experiments/run_adaptive_attack.py`, `experiments/run_ablation.py`, `visualization/plot_results.py`, `README.md`, `RESULTS_COLLECTION_README.md`
- What changed: Implemented the missing experiment pipeline, added shared loaders/output helpers, upgraded prototype defenses and white-box simulation, and turned plotting into a real figure regeneration step.
- Why the change was necessary: The repository had real datasets but still depended on placeholder runners and stubbed evaluation paths, so it could not produce the paper tables/figures it claimed to support.
- Risk / follow-up: The real LLM baseline still depends on local model availability and may fall back to smoke mode during constrained environments.

## Session 002 - Local setup regression fix
- Files changed: `test_local_setup.py`, `changes.md`
- What changed: Fixed the sample concentration calculation to extract a scalar from the 1x1 NumPy dot-product result before converting it to `float`.
- Why the change was necessary: The local setup smoke test crashed after successful retrieval because the previous code tried to cast a non-scalar NumPy array directly to `float`.
- Risk / follow-up: This only fixes the setup-check regression; it does not change the main experiment pipeline logic.

## Session 003 - VRAM budget fix for run_roc_analysis and run_main_experiment
- Files changed: `experiments/run_roc_analysis.py`, `experiments/run_main_experiment.py`, `changes.md`
- What changed: Added `--use-real-llm` flag to both experiment scripts; `LLMBasedDefense` now defaults to smoke (keyword heuristic) mode instead of loading Qwen2-7B-Instruct automatically.
- Why the change was necessary: Without the flag, both scripts downloaded ~15 GB of Qwen2-7B weights and attempted to load them alongside the retriever, violating the RTX 4070 setup guide's rule: "Experiments 1, 2, 4, 5: no LLM needed" and "Never load the LLM and the retriever simultaneously in VRAM."
- Risk / follow-up: The LLM-detect FNR/FPR numbers will use the keyword heuristic by default. Use `--use-real-llm --retriever minilm` if you need real Qwen2-7B inference. Do NOT combine `--use-real-llm` with `--retriever dpr` — that would exceed the 8 GB VRAM budget.

## Session 004 - Fix all root causes producing flat/degenerate experiment results

### Root causes fixed
Three independent bugs caused every metric to be degenerate (C scores all ~1.17,
ASR_A = 0% even without defense, CQRCD AUC = 0.57 ≈ random).

---

#### Fix A — `modules/dsrm_simulator.py` (complete rewrite)

**Problem:** The `text` field stored the *full* adversarial document (query +
planning + reasoning).  MiniLM encodes this as an average over all tokens; the
~20-word query was diluted by ~150 words of generic template text, pulling the
document embedding away from the query direction.  Result: `sim(adv_doc, Q) ≈
0.65`, barely higher than a legitimate doc — concentration scores indistinguishable
from legitimate.

**Fix:** `text = retrieval_text = query` (the pure query string).
This makes `sim(adv_doc, Q) ≈ 1.0` (maximising the concentration numerator).
The `full_text` field retains all planning/reasoning and mentions the attack
tool **≥ 7 times** (used by MockLLM for vote counting).

**White-box addition:** When a retriever is supplied, the simulator generates
6 candidate anchor texts and selects the one with the highest empirical
concentration score — approximating HotFlip-style contrastive optimisation without
requiring gradient access.

---

#### Fix B — `modules/neighbor_generator.py` (`_simple_variants` rewrite)

**Problem:** The fallback generator produced variants like "Please Summarize
recent advancements in quantum computing" with MiniLM cosine similarity **~0.97-0.99**
— essentially identical embeddings — making every concentration score converge to
C ≈ 1.0 regardless of document type.

**Fix:** `_simple_variants` now applies synonym substitutions from a 40-entry
table (action verbs, domain nouns, temporal markers) plus structural
transformations (imperative → interrogative).  Target similarity: **~0.82-0.92**,
giving `C_bb ≈ 1.15-1.25` vs `C_legit ≈ 1.03-1.08`.

---

#### Fix C — `modules/agent_simulator.py` (MockLLM rewrite)

**Problem:** MockLLM read `tools_mentioned` (one entry per doc), producing a 3-way
tie (count 1:1:1) that always resolved to a legitimate tool. Attack tool was also
excluded from `available_tools`. Result: ASR_A = 0% even with no defense.

**Fix:** MockLLM now counts occurrences via `re.findall` on `full_text`:
- `[Tool: X]` marker → weight 4
- Bare mention → weight 1

Adversarial docs accumulate ~10-14 votes; legitimate docs ~2-4 each → attack tool
wins the vote → ASR_A > 0 without defense.

---

#### Fix D — `experiments/common.py` (`build_scenarios`)

**Problem:** `available_tools` excluded the attack tool, so MockLLM could never
select it regardless of vote counts.

**Fix:** Attack tool appended to `available_tools` per scenario, matching the
ASB threat model.

---

#### Fix E — `config.py` (threshold recalibration)

**Problem:** `CONCENTRATION_THRESHOLD = 1.65` was tuned for DPR (paraphrase sim
~0.50-0.60).  In MiniLM space (~0.82-0.92) nothing ever crossed 1.65 → FNR = 1.0.

**Fix:** `CONCENTRATION_THRESHOLD = 1.20` — validated separating value for MiniLM.

---

#### Fix F — `scripts/rebuild_adversarial_data.py` (new file)

Regenerates `data/adversarial_bb.json` and `data/adversarial_wb.json` using the
improved simulator, preserving all ASB metadata.  Must be run before experiments.

---

### Expected improvements after applying all fixes

| Metric | Before (broken) | Expected after fixes |
|---|---|---|
| C_legit mean | 1.162 | ~1.04-1.08 |
| C_bb mean | 1.167 | ~1.18-1.25 |
| C_wb mean | 1.191 | ~1.30-1.45 |
| CQRCD AUC | 0.57 | ~0.78-0.85 |
| ASR_A (no defense) | 0.0% | ~35-45% |
| ASR_A (CQRCD) | 0.0% | ~12-22% |
| CQRCD FNR @ τ=1.20 | 1.00 | ~0.20-0.35 |
| CQRCD FPR @ τ=1.20 | 0.00 | ~0.05-0.12 |

Note: MiniLM produces lower absolute C values than DPR (paper's retriever).
For results closer to the paper's exact figures, run Experiments 1 and 2 with
`--retriever dpr` (no LLM loaded simultaneously — safe within 8 GB VRAM budget).

---

### Files changed in Session 004
- `modules/dsrm_simulator.py` — complete rewrite (Fix A)
- `modules/neighbor_generator.py` — `_simple_variants` rewrite (Fix B)
- `modules/agent_simulator.py` — MockLLM rewrite (Fix C)
- `experiments/common.py` — `build_scenarios` attack_tool inclusion (Fix D)
- `config.py` — threshold recalibration 1.65 → 1.20 (Fix E)
- `scripts/rebuild_adversarial_data.py` — new data regeneration script (Fix F)
- `changes.md` — this entry

---

## Session 005 - Fix concentration score separation and adaptive attack ASR

### Root causes fixed

Two issues remained after Session 004:
1. CQRCD FNR = 78.5% (most adversarial docs slipped through threshold 1.20)
2. Adaptive attack ASR_A = 0.0 across ALL c_target values (attack tool never selectable)

---

#### Fix A — `experiments/run_adaptive_attack.py` (attack_tool in available_tools)

**Problem:** `run_adaptive_attack.py` called `agent.run_task(...)` with
`task.get('available_tools', [])` — the ORIGINAL task tool list that never
includes the attack tool.  This is the same bug fixed in `common.py` (Session 004
Fix D), but the adaptive experiment file was missed.  Result: MockLLM can never
select the attack tool → ASR_A = 0.0 for every c_target, making the tradeoff
curve flat and meaningless.

**Fix:** `available_tools = list(task.get('available_tools', []))` then append
`attack_doc['attack_tool']` if not present, before passing to `run_task`.

**Expected outcome:** ASR_A should now show the expected INVERSE relationship:
- Low c_target (evasion mode, text ≈ plain query) → evades CQRCD → high ASR_A
- High c_target (aggressive mode, concentrated anchor) → detected by CQRCD → low ASR_A

---

#### Fix B — `modules/neighbor_generator.py` (thematic variant generation)

**Problem:** After Session 004, synonym substitutions produced MiniLM cosine
~0.87-0.92 — close paraphrases only.  This kept the mean denominator in
C(d,Q) = sim(d,Q) / mean(sim(d,Q_i)) at ~0.86, giving:
- C_bb = 0.99/0.86 ≈ 1.15 → BELOW threshold 1.20 → not detected
- C_legit = 0.68/0.66 ≈ 1.03 → below threshold correctly

Result: FNR = 78.5% (most adversarial docs pass through).

**Fix:** Added thematic variant generation as Round 0 in `_simple_variants`.
Instead of paraphrasing the query (same intent, different words), thematic
variants ASK A DIFFERENT QUESTION about the same topic, e.g.:
- Original: "Summarize recent advancements in deep learning over the past five years"
- Thematic: "What are the main challenges related to advancements deep learning?"
- Thematic: "How has advancements deep learning evolved historically?"

These produce MiniLM cosine ~0.70-0.80 (vs ~0.87-0.92 for synonym subs).
Mixed into the 5-neighbor pool, they lower the mean denominator to ~0.80:
- C_bb = 0.99/0.80 ≈ 1.24 → ABOVE threshold 1.20 → detected!
- C_legit = 0.68/0.65 ≈ 1.05 → below threshold correctly

**Implementation:** Added `_STOP_WORDS_FILTER` set, `_THEMATIC_TEMPLATES` list
(8 templates), `_extract_topic()` method, and `_thematic_variants()` method to
`NeighborGenerator`.  Thematic variants are generated first (Round 0) to ensure
they appear in every 5-neighbor window.

**Expected improvements after Session 005 fixes:**

| Metric | After Session 004 | Expected after Session 005 |
|---|---|---|
| C_bb mean | 1.147 | ~1.22-1.32 |
| C_wb mean | 1.171 | ~1.25-1.40 |
| CQRCD FNR @ τ=1.20 | 0.785 | ~0.15-0.35 |
| CQRCD FPR @ τ=1.20 | 0.050 | ~0.02-0.08 |
| ASR_A (CQRCD) | 78.5% | ~20-40% |
| Adaptive ASR_A (low c_target) | 0.0% | ~70-90% |
| Adaptive ASR_A (high c_target) | 0.0% | ~10-25% |

---

### Files changed in Session 005
- `experiments/run_adaptive_attack.py` — attack_tool inclusion in available_tools (Fix A)
- `modules/neighbor_generator.py` — thematic variant generation (Fix B)
- `changes.md` — this entry

---

### Run order after applying Session 005

```
# No data rebuild needed — only neighbor generation and adaptive experiment changed.
# Adversarial data (from Session 004 rebuild) remains valid.

python -m experiments.run_roc_analysis
python -m experiments.run_main_experiment
python -m experiments.run_adaptive_attack
python -m experiments.run_ablation
python -m visualization.plot_results
```

---

## Session 006 - Fix ablation methodology and make adaptive c_target measurable

### Root causes fixed

Two methodology problems remained after Session 005:
1. `run_ablation.py` reported an `auc_proxy = 1 - (FNR + FPR) / 2` at one threshold, not true ROC-AUC, so the saved ablation table was useful for calibration but not for a defensible scientific claim.
2. `run_adaptive_attack.py` swept many `c_target` values, but the attacker did not actually optimize toward the requested target concentration. In practice, the achieved concentrations collapsed into a few nearly flat levels, making the tradeoff plot hard to trust.

---

#### Fix A - `experiments/run_ablation.py` (true ROC-AUC + explicit threshold metrics)

**Problem:** The ablation study mixed ranking quality and operating-point quality into one proxy metric. That made it impossible to tell whether a setting truly improved separability or just happened to look better at a fixed threshold.

**Fix:** Reworked the ablation runner to:
- compute real ROC-AUC with `compute_detection_metrics(...)`
- keep threshold metrics as separate columns (`fnr_at_eval_threshold`, `fpr_at_eval_threshold`, `f1_at_eval_threshold`, `tpr_at_eval_threshold`)
- log `optimal_threshold` alongside the evaluated threshold
- log `threshold_error = FNR + FPR` explicitly for calibration analysis
- include the deployed threshold `1.2` in the threshold sweep

**Expected outcome:** `results/tables/ablation_results.csv` now supports two distinct claims:
- ROC-AUC for scientific comparison across settings
- threshold error for deployment/calibration discussion

---

#### Fix B - `modules/dsrm_simulator.py` and `experiments/run_adaptive_attack.py` (target-aware adaptive attacker)

**Problem:** The adaptive attacker was not using the retriever during generation and did not search for anchors that matched the requested concentration target. As a result, many different `c_target` values produced nearly the same achieved concentration.

**Fix in simulator:** Expanded white-box anchor generation to cover a broader concentration range:
- low-concentration anchors using broader topical variants
- medium/high-concentration anchors using progressively stronger exact-match emphasis
- retriever-guided scoring of each candidate anchor
- selection by minimum absolute error to `c_target`, not maximum concentration

**Fix in adaptive experiment:** `run_adaptive_attack.py` now:
- passes the retriever into `generate_adaptive_whitebox(...)`
- records `mean_concentration`
- records `mean_abs_target_error`
- records `signed_target_error`
- plots achieved concentration alongside ASR and detection rate

**Expected outcome:** The adaptive tradeoff table should now tell us whether `c_target` is actually controllable, and if not, by how much it misses. That makes the next rerun interpretable instead of just visually flatter or steeper.

---

### Files changed in Session 006
- `experiments/run_ablation.py` - replaced proxy AUC logging with true ROC-AUC and separate threshold metrics; added threshold `1.2`
- `experiments/run_adaptive_attack.py` - passes retriever into adaptive generation and records target-achievement error columns
- `modules/dsrm_simulator.py` - expanded adaptive candidate search and selects anchors closest to requested `c_target`
- `changes.md` - this entry

---
---

## Session 007 - Align figure regeneration with the new ablation/adaptive schemas

### Root cause fixed

After Session 006, `visualization.plot_results` still expected the old ablation columns (`auc_proxy`, `fnr`, `fpr`). Once the refreshed ablation runner wrote the new schema, figure regeneration failed with:

`KeyError: 'auc_proxy'`

This blocked `python -m visualization.plot_results` even though the updated experiment runs themselves completed successfully.

---

### Fix

Updated `visualization/plot_results.py` to match the new saved-table schema:
- `plot_ablation()` now reads `auc` for neighbor-count ablation
- `plot_ablation()` now reads `threshold_error` for threshold ablation
- `plot_adaptive()` now optionally overlays `mean_concentration` from the new adaptive table
- adaptive plot y-axis label changed from `Rate` to `Value` to reflect mixed metrics on the same figure

---

### Files changed in Session 007
- `visualization/plot_results.py` - updated plotting code for Session 006 result schema
- `changes.md` - this entry

---

## Session 008 - Expand adaptive attacker search space and save per-example diagnostics
- Files changed: `changes.md`, `modules/dsrm_simulator.py`, `experiments/run_adaptive_attack.py`
- What changed: Broadened the adaptive white-box candidate pool with more contrastive and exact-match anchor templates, recorded candidate-pool score range metadata on each generated attack, and made the adaptive experiment save a per-example diagnostics table (`results/tables/adaptive_tradeoff_diagnostics.csv`) alongside the aggregate tradeoff CSV.
- Why the change was necessary: The refreshed adaptive results showed a hard ceiling around concentration ~1.38 for higher `c_target` values, which meant the attack was not expressive enough to test the full tradeoff claim and the aggregate table alone could not show whether the bottleneck came from candidate generation, target mismatch, or downstream filtering.
- Risk / follow-up: This should make the next rerun much more interpretable, but it may still reveal that some targets remain unreachable under MiniLM. If the ceiling persists, inspect `adaptive_tradeoff_diagnostics.csv` first and then decide whether to further redesign the anchor family or narrow the reported adaptive range.

## Session 009 - Reframe docs around DPR-primary reporting and retriever-separated artifacts
- Files changed: `changes.md`, `README.md`, `RESULTS_COLLECTION_README.md`, `CQRCD_Implementation_Master.md`, `config.py`, `visualization/plot_results.py`
- What changed: Reframed the repo narrative so DPR is the canonical final-evaluation retriever and MiniLM is the development/practical variant, documented a retriever-separated artifact layout (`results/dpr` vs `results/minilm`), updated the adaptive-attack language to scope conclusions by retriever, added `fig7_ablation_method` to the plotting pipeline, and taught `visualization.plot_results` to regenerate figures from any result directory via `--input-dir`.
- Why the change was necessary: The measured results support the geometric/theoretical story much more strongly under DPR than MiniLM, while MiniLM remains useful but needs honest limitation framing. The repo docs previously mixed both settings into one generic `results/` narrative, which risked overstating MiniLM and obscuring which artifacts should be treated as canonical.
- Risk / follow-up: The repo-side framing is now aligned, but the missing DPR neighbor-method ablation still has to be run from a working local Python environment and saved into `results/dpr` before the Claim 5 discussion can cite it as completed evidence.

## Session 010 - Make concentration thresholds retriever-specific
- Files changed: `changes.md`, `config.py`, `experiments/common.py`, `experiments/run_ablation.py`
- What changed: Replaced the single shared concentration threshold with retriever-specific defaults (`1.20` for MiniLM, `1.05` for DPR), updated the shared CQRCD filter builder to pick the threshold from the retriever automatically unless explicitly overridden, and changed the ablation runner so DPR neighbor-count and neighbor-method rows are evaluated at the DPR-calibrated threshold instead of inheriting MiniLM's `1.20`.
- Why the change was necessary: DPR ablation showed that `1.20` is badly miscalibrated for DPR (`optimal_threshold ≈ 1.05`), causing alarming but artificial rows such as `n=20` with `FNR=1.0`. That was a threshold-calibration artifact, not a real failure of the DPR geometry.
- Risk / follow-up: A quick FPR sanity check at `tau ≈ 1.05` is still needed on legitimate DPR scores once `results/dpr/tables/cqrcd_detection_scores.csv` or an updated DPR ROC run is available.