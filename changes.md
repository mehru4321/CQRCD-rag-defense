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

