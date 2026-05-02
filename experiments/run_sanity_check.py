"""Experiment 1: concentration-score separation on the real dataset."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import (
    build_cqrcd_filter,
    ensure_output_dirs,
    init_retriever,
    load_dataset_bundle,
    sample_for_smoke,
    save_dataframe,
    save_figure,
    summarize_group,
)


def score_documents(filter_model, docs, label):
    rows = []
    for doc in docs:
        query = doc.get('target_query') or doc.get('source_query') or doc.get('text', '')
        score = filter_model.compute_concentration_score(doc.get('text', doc.get('full_text', '')), query)
        rows.append(
            {
                'doc_id': doc['id'],
                'group': label,
                'query': query,
                'score': score,
            }
        )
    return rows


def build_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    for group in ['legitimate', 'blackbox', 'whitebox']:
        subset = df[df['group'] == group]['score']
        ax.hist(subset, bins=20, alpha=0.5, label=group, density=True)
    ax.set_title('CQRCD concentration score distributions')
    ax.set_xlabel('Concentration score')
    ax.set_ylabel('Density')
    ax.legend()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Experiment 1 concentration sanity check.')
    parser.add_argument('--retriever', default='minilm', choices=['minilm', 'dpr'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    outputs = ensure_output_dirs(args.output_dir)
    bundle = load_dataset_bundle()

    legitimate = bundle['legitimate']
    blackbox = bundle['blackbox']
    whitebox = bundle['whitebox']
    if args.smoke:
        legitimate = sample_for_smoke(legitimate, limit=20)
        blackbox = sample_for_smoke(blackbox, limit=40)
        whitebox = sample_for_smoke(whitebox, limit=40)

    retriever = init_retriever(args.retriever)
    filter_model = build_cqrcd_filter(retriever, smoke=args.smoke)

    score_rows = []
    score_rows.extend(score_documents(filter_model, legitimate, 'legitimate'))
    score_rows.extend(score_documents(filter_model, blackbox, 'blackbox'))
    score_rows.extend(score_documents(filter_model, whitebox, 'whitebox'))

    scores_df = pd.DataFrame(score_rows)
    summary_df = pd.DataFrame(
        [
            summarize_group(scores_df[scores_df['group'] == 'legitimate']['score'], 'legitimate'),
            summarize_group(scores_df[scores_df['group'] == 'blackbox']['score'], 'blackbox'),
            summarize_group(scores_df[scores_df['group'] == 'whitebox']['score'], 'whitebox'),
        ]
    )

    save_dataframe(summary_df, outputs['tables'] / 'concentration_summary.csv')
    save_dataframe(scores_df, outputs['tables'] / 'concentration_scores.csv')
    save_figure(build_plot(scores_df), outputs['figures'] / 'fig1_concentration_dist')

    print('Saved concentration summary and score distributions.')


if __name__ == '__main__':
    main()
