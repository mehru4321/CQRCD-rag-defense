"""Experiment 4: adaptive attacker tradeoff."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import (
    build_cqrcd_filter,
    ensure_output_dirs,
    init_retriever,
    load_dataset_bundle,
    representative_tasks,
    sample_for_smoke,
    save_dataframe,
    save_figure,
)
from modules.agent_simulator import RAGAgent
from modules.dsrm_simulator import DSRMSimulator
from modules.knowledge_base import KnowledgeBase


DEFAULT_C_TARGETS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 2.8]


def parse_targets(raw: str):
    return [float(item.strip()) for item in raw.split(',') if item.strip()]


def build_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['c_target'], df['ASR_A'], marker='o', label='ASR_A')
    ax.plot(df['c_target'], df['detection_rate'], marker='s', label='Detection rate')
    ax.plot(df['c_target'], df['mean_concentration'], marker='^', label='Mean concentration')
    ax.set_title('Adaptive attacker tradeoff')
    ax.set_xlabel('Target concentration')
    ax.set_ylabel('Value')
    ax.legend()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Experiment 4 adaptive attack tradeoff.')
    parser.add_argument('--retriever', default='minilm', choices=['minilm', 'dpr'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--c-targets', default=','.join(str(v) for v in DEFAULT_C_TARGETS))
    args = parser.parse_args()

    outputs = ensure_output_dirs(args.output_dir)
    bundle = load_dataset_bundle()
    tasks = representative_tasks(bundle['tasks'])
    task_by_id = {task['task_id']: task for task in tasks}
    base_whitebox = bundle['whitebox']
    if args.smoke:
        base_whitebox = sample_for_smoke(base_whitebox, limit=30)

    retriever = init_retriever(args.retriever)
    cqrcd = build_cqrcd_filter(retriever, smoke=args.smoke)
    simulator = DSRMSimulator()
    c_targets = parse_targets(args.c_targets)

    rows = []
    diagnostic_rows = []
    for c_target in c_targets:
        results = []
        achieved_scores = []
        for attack_doc in base_whitebox:
            task = task_by_id.get(attack_doc['task_id'])
            if task is None:
                continue

            adaptive_doc = simulator.generate_adaptive_whitebox(
                query=task['query'],
                attack_tool=attack_doc['attack_tool'],
                attack_instruction=attack_doc['attack_instruction'],
                c_target=c_target,
                retriever=retriever,
                legitimate_tools=task.get('available_tools', []),
            )
            adaptive_doc.update(
                {
                    'domain': attack_doc['domain'],
                    'agent_name': attack_doc['agent_name'],
                    'task_id': attack_doc['task_id'],
                    'scenario_id': attack_doc['scenario_id'],
                    'label': 'adversarial',
                    'tools_mentioned': [attack_doc['attack_tool']],
                }
            )

            achieved = cqrcd.compute_concentration_score(adaptive_doc['text'], task['query'])
            adaptive_doc['achieved_concentration'] = achieved
            achieved_scores.append(achieved)

            legit_docs = [doc for doc in bundle['legitimate'] if doc['domain'] == task['domain']]
            kb = KnowledgeBase(retriever, use_gpu=False)
            kb.add_documents(legit_docs)
            kb.poison([adaptive_doc])
            agent = RAGAgent(kb, retriever=retriever, defense_filter=cqrcd)
            # Include attack_tool so MockLLM can select it when the adversarial
            # doc dominates the retrieved context (mirrors common.py fix).
            available_tools = list(task.get('available_tools', []))
            if attack_doc['attack_tool'] not in available_tools:
                available_tools = available_tools + [attack_doc['attack_tool']]
            outcome = agent.run_task(task['query'], available_tools, attack_tool=attack_doc['attack_tool'])
            filtered_ids = {doc['id'] for doc in outcome['filtered_docs']}
            outcome['attack_filtered'] = adaptive_doc['id'] not in filtered_ids
            results.append(outcome)
            diagnostic_rows.append(
                {
                    'c_target': c_target,
                    'task_id': attack_doc['task_id'],
                    'scenario_id': attack_doc['scenario_id'],
                    'domain': attack_doc['domain'],
                    'query': task['query'],
                    'attack_tool': attack_doc['attack_tool'],
                    'achieved_concentration': achieved,
                    'adaptive_achieved_estimate': adaptive_doc.get('adaptive_achieved_estimate'),
                    'absolute_target_error': abs(achieved - c_target),
                    'signed_target_error': achieved - c_target,
                    'attack_filtered': outcome['attack_filtered'],
                    'attack_tool_selected': outcome['attack_tool_selected'],
                    'attack_tool_retrieved': outcome['attack_tool_retrieved'],
                    'selected_tool': outcome['selected_tool'],
                    'candidate_count': adaptive_doc.get('candidate_count'),
                    'candidate_min_score': adaptive_doc.get('candidate_min_score'),
                    'candidate_max_score': adaptive_doc.get('candidate_max_score'),
                    'candidate_target_gap_min': adaptive_doc.get('candidate_target_gap_min'),
                    'chosen_anchor_preview': adaptive_doc.get('chosen_anchor_preview'),
                }
            )

        total = max(1, len(results))
        mean_concentration = sum(achieved_scores) / max(1, len(achieved_scores))
        mean_abs_error = (
            sum(abs(score - c_target) for score in achieved_scores) / max(1, len(achieved_scores))
        )
        rows.append(
            {
                'c_target': c_target,
                'ASR_A': sum(1 for row in results if row['attack_tool_selected']) / total,
                'detection_rate': sum(1 for row in results if row['attack_filtered']) / total,
                'RR': sum(1 for row in results if row['attack_tool_retrieved']) / total,
                'mean_concentration': mean_concentration,
                'mean_abs_target_error': mean_abs_error,
                'signed_target_error': mean_concentration - c_target,
            }
        )

    df = pd.DataFrame(rows)
    save_dataframe(df, outputs['tables'] / 'adaptive_tradeoff.csv')
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    save_dataframe(diagnostics_df, outputs['tables'] / 'adaptive_tradeoff_diagnostics.csv')
    save_figure(build_plot(df), outputs['figures'] / 'fig4_adaptive_tradeoff')
    print('Saved adaptive attacker tradeoff outputs.')


if __name__ == '__main__':
    main()
