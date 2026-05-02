from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.dsrm_simulator import DSRMSimulator


# Fields that the DSRM simulator owns — everything else is preserved from the
# original ASB-sourced record.
_DSRM_FIELDS = {
    'text', 'retrieval_text', 'planning_text', 'reasoning_text',
    'full_text', 'tools_mentioned', 'concentration_score',
    'adaptive_target', 'adaptive_strength',
}


def _load_json(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def _save_json(data, path: Path) -> None:
    with path.open('w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    print(f"  Saved {len(data)} records → {path}")


def _build_task_tools(tasks: list) -> dict:
    """Return {task_id: [available_tools]} from asb_tasks.json."""
    return {t['task_id']: t.get('available_tools', []) for t in tasks}


def rebuild_blackbox(docs: list, sim: DSRMSimulator, task_tools: dict) -> list:
    rebuilt = []
    for i, doc in enumerate(docs):
        task_id = doc.get('task_id', '')
        query = doc.get('target_query') or doc.get('retrieval_text', '')
        attack_tool = doc.get('attack_tool', '')
        attack_instruction = doc.get('attack_instruction', '')
        legitimate_tools = task_tools.get(task_id, [])

        new_fields = sim.generate_blackbox(
            query=query,
            attack_tool=attack_tool,
            attack_instruction=attack_instruction,
            legitimate_tools=legitimate_tools,
        )

        patched = {k: v for k, v in doc.items() if k not in _DSRM_FIELDS}
        for field in _DSRM_FIELDS:
            if field in new_fields:
                patched[field] = new_fields[field]
        # Restore original id and domain from ASB record
        patched['id'] = doc['id']
        patched['domain'] = doc.get('domain', new_fields.get('domain', 'unknown'))
        patched['attack_mode'] = 'blackbox'
        rebuilt.append(patched)

        if (i + 1) % 50 == 0:
            print(f"    blackbox: {i + 1}/{len(docs)}")
    return rebuilt


def rebuild_whitebox(
    docs: list,
    sim: DSRMSimulator,
    task_tools: dict,
    retriever=None,
) -> list:
    rebuilt = []
    for i, doc in enumerate(docs):
        task_id = doc.get('task_id', '')
        query = doc.get('target_query') or doc.get('retrieval_text', '')
        attack_tool = doc.get('attack_tool', '')
        attack_instruction = doc.get('attack_instruction', '')
        legitimate_tools = task_tools.get(task_id, [])
        c_target = doc.get('adaptive_target')

        new_fields = sim.generate_whitebox(
            query=query,
            attack_tool=attack_tool,
            attack_instruction=attack_instruction,
            retriever=retriever,
            legitimate_tools=legitimate_tools,
            c_target=c_target,
        )

        patched = {k: v for k, v in doc.items() if k not in _DSRM_FIELDS}
        for field in _DSRM_FIELDS:
            if field in new_fields:
                patched[field] = new_fields[field]
        patched['id'] = doc['id']
        patched['domain'] = doc.get('domain', new_fields.get('domain', 'unknown'))
        patched['attack_mode'] = 'whitebox'
        rebuilt.append(patched)

        if (i + 1) % 50 == 0:
            print(f"    whitebox: {i + 1}/{len(docs)}")
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(description='Rebuild adversarial data files.')
    parser.add_argument('--data-dir', default='data', help='Path to data/ directory')
    parser.add_argument(
        '--use-retriever',
        action='store_true',
        help='Load MiniLM retriever for white-box anchor optimisation (recommended)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print first 3 rebuilt docs per file without saving',
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    bb_path = data_dir / 'adversarial_bb.json'
    wb_path = data_dir / 'adversarial_wb.json'
    tasks_path = data_dir / 'asb_tasks.json'

    print("Loading source files …")
    bb_docs = _load_json(bb_path)
    wb_docs = _load_json(wb_path)
    tasks = _load_json(tasks_path)
    task_tools = _build_task_tools(tasks)
    print(f"  {len(bb_docs)} black-box docs, {len(wb_docs)} white-box docs, "
          f"{len(tasks)} tasks")

    retriever = None
    if args.use_retriever:
        print("Loading MiniLM retriever for white-box optimisation …")
        try:
            from modules.retriever import DenseRetriever
            retriever = DenseRetriever('minilm')
            print("  Retriever loaded.")
        except Exception as exc:
            print(f"  WARNING: could not load retriever ({exc}). "
                  "Falling back to heuristic white-box anchor.")

    sim = DSRMSimulator(seed=42)

    print("Rebuilding black-box documents …")
    new_bb = rebuild_blackbox(bb_docs, sim, task_tools)

    print("Rebuilding white-box documents …")
    new_wb = rebuild_whitebox(wb_docs, sim, task_tools, retriever=retriever)

    if args.dry_run:
        print("\n--- DRY RUN: first 3 black-box docs ---")
        for doc in new_bb[:3]:
            print(json.dumps({
                'id': doc['id'],
                'text': doc['text'][:80],
                'full_text_preview': doc['full_text'][:120],
                'tools_mentioned': doc['tools_mentioned'],
            }, indent=2))
        print("\n--- DRY RUN: first 3 white-box docs ---")
        for doc in new_wb[:3]:
            print(json.dumps({
                'id': doc['id'],
                'text': doc['text'][:120],
                'adaptive_strength': doc.get('adaptive_strength'),
            }, indent=2))
        print("\nDry run complete — no files written.")
        return

    print("Saving …")
    _save_json(new_bb, bb_path)
    _save_json(new_wb, wb_path)
    print("Done.  Re-run all experiments to generate updated results.")


if __name__ == '__main__':
    main()
