# CQRCD Data Directory

Expected files follow the formats in `CQRCD_Implementation_Master.md`:

- `asb_tasks.json`
- `legitimate_kb.json`
- `adversarial_bb.json`
- `adversarial_wb.json`

Generated adversarial files should be reproducible with `RANDOM_SEED = 42`.

Raw ASB files belong in `data/raw_asb/`:

- `agent_task.jsonl`
- `all_normal_tools.jsonl`
- `all_attack_tools.jsonl`

Generate processed CQRCD files with:

```bash
python -m scripts.prepare_asb_data
```

By default this creates the master-plan 400 attack scenarios: one representative
task per agent/domain times 40 attack tools per agent. Use `--all-tasks` to
generate the larger 2,000-document-per-attack-mode variant for all 50 ASB tasks.

`data/raw_asb/` and `data/adversarial*.json` are generated/local data and are
ignored by git. Re-run the preparation script whenever those files need to be
recreated.
