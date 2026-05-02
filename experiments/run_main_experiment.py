"""Experiment 3: attack success rate under different defenses."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import compute_asr_a, compute_asr_r, compute_retrieval_rate
from experiments.common import (
    build_cqrcd_filter,
    build_scenarios,
    ensure_output_dirs,
    init_retriever,
    load_dataset_bundle,
    representative_tasks,
    sample_for_smoke,
    save_dataframe,
    save_figure,
)
from modules.agent_simulator import RAGAgent
from modules.baseline_defenses import LLMBasedDefense, PerplexityDefense
from modules.knowledge_base import KnowledgeBase


def evaluate_scenarios(scenarios, retriever, defense_name, defense_obj=None):
    results = []
    for scenario in scenarios:
        kb = KnowledgeBase(retriever, use_gpu=False)
        kb.add_documents(scenario['legitimate_docs'])
        kb.poison([scenario['attack_doc']])
        agent = RAGAgent(
            knowledge_base=kb,
            retriever=retriever,
            defense_filter=defense_obj,
            k=5,
        )
        outcome = agent.run_task(
            scenario['query'],
            available_tools=scenario['available_tools'],
            attack_tool=scenario['attack_tool'],
        )
        filtered_ids = {doc['id'] for doc in outcome['filtered_docs']}
        attack_filtered = scenario['attack_doc']['id'] not in filtered_ids
        legit_ids = {doc['id'] for doc in scenario['legitimate_docs']}
        removed_legit = sum(1 for doc in outcome['retrieved_docs'] if doc['id'] in legit_ids and doc['id'] not in filtered_ids)
        legit_in_retrieved = sum(1 for doc in outcome['retrieved_docs'] if doc['id'] in legit_ids)
        outcome.update(
            {
                'scenario_id': scenario['scenario_id'],
                'defense': defense_name,
                'attack_filtered': attack_filtered,
                'legitimate_removed': removed_legit,
                'legitimate_retrieved': legit_in_retrieved,
            }
        )
        results.append(outcome)

    return {
        'defense': defense_name,
        'results': results,
        'ASR_A': compute_asr_a(results),
        'ASR_R': compute_asr_r(results),
        'RR': compute_retrieval_rate(results),
        'detection_rate': sum(1 for row in results if row['attack_filtered']) / max(1, len(results)),
        'FPR': sum(row['legitimate_removed'] for row in results) / max(1, sum(row['legitimate_retrieved'] for row in results)),
    }


def build_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df['defense'], df['ASR_A'], color=['#7f8c8d', '#e67e22', '#3498db', '#2ecc71'])
    ax.set_title('Attack success under defense conditions')
    ax.set_ylabel('ASR_A')
    ax.set_ylim(0, max(0.05, float(df['ASR_A'].max()) * 1.2))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Experiment 3 main ASR experiment.')
    parser.add_argument('--retriever', default='minilm', choices=['minilm', 'dpr'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--use-real-llm', action='store_true',
                        help='Load Qwen2-7B-Instruct for the LLM baseline (~4.2 GB VRAM). '
                             'Only use with --retriever minilm. Defaults to smoke (keyword) mode '
                             'to stay within the 8 GB VRAM budget per the RTX 4070 setup guide.')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--defense', default='all', choices=['none', 'ppl', 'llm', 'cqrcd', 'all'])
    parser.add_argument('--attack-mode', default='both', choices=['blackbox', 'whitebox', 'both'])
    args = parser.parse_args()

    outputs = ensure_output_dirs(args.output_dir)
    bundle = load_dataset_bundle()
    tasks = representative_tasks(bundle['tasks'])

    attack_docs = []
    if args.attack_mode in ('blackbox', 'both'):
        attack_docs.extend(bundle['blackbox'])
    if args.attack_mode in ('whitebox', 'both'):
        attack_docs.extend(bundle['whitebox'])
    if args.smoke:
        attack_docs = sample_for_smoke(attack_docs, limit=60)

    scenarios = build_scenarios(tasks, bundle['legitimate'], attack_docs)
    retriever = init_retriever(args.retriever)

    defenses = []
    if args.defense in ('none', 'all'):
        defenses.append(('none', None))
    if args.defense in ('ppl', 'all'):
        defenses.append(('ppl', PerplexityDefense()))
    if args.defense in ('llm', 'all'):
        llm_mode = 'real' if (args.use_real_llm and not args.smoke) else 'smoke'
        defenses.append(('llm', LLMBasedDefense(mode=llm_mode)))
    if args.defense in ('cqrcd', 'all'):
        defenses.append(('cqrcd', build_cqrcd_filter(retriever, smoke=args.smoke)))

    aggregate_rows = []
    detail_rows = []
    for defense_name, defense_obj in defenses:
        report = evaluate_scenarios(scenarios, retriever, defense_name, defense_obj=defense_obj)
        aggregate_rows.append({key: value for key, value in report.items() if key != 'results'})
        detail_rows.extend(report['results'])

    aggregate_df = pd.DataFrame(aggregate_rows)
    details_df = pd.DataFrame(detail_rows)

    save_dataframe(aggregate_df, outputs['tables'] / 'asr_results.csv')
    save_dataframe(details_df, outputs['tables'] / 'asr_results_detailed.csv')
    save_figure(build_plot(aggregate_df), outputs['figures'] / 'fig3_asr_bar')

    print('Saved ASR results and defense comparison figure.')


if __name__ == '__main__':
    main()
