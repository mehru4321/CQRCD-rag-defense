"""Experiment 5: ablation study for core CQRCD settings."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import compute_threshold_metrics
from experiments.common import (
    build_cqrcd_filter,
    ensure_output_dirs,
    init_retriever,
    load_dataset_bundle,
    sample_for_smoke,
    save_dataframe,
    save_figure,
)


def gather_scores(filter_model, docs):
    scores = []
    labels = []
    for doc in docs:
        query = doc.get('target_query') or doc.get('source_query') or doc.get('text', '')
        scores.append(filter_model.compute_concentration_score(doc.get('text', doc.get('full_text', '')), query))
        labels.append(0 if doc.get('label') == 'legitimate' else 1)
    return labels, scores


def build_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df[x_col], df[y_col], marker='o')
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Experiment 5 ablation study.')
    parser.add_argument('--retriever', default='minilm', choices=['minilm', 'dpr'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    outputs = ensure_output_dirs(args.output_dir)
    bundle = load_dataset_bundle()
    docs = bundle['legitimate'] + bundle['blackbox'] + bundle['whitebox']
    if args.smoke:
        docs = sample_for_smoke(docs, limit=100)

    retriever = init_retriever(args.retriever)
    rows = []

    for n_neighbors in [1, 3, 5, 10, 20]:
        filter_model = build_cqrcd_filter(retriever, smoke=args.smoke, n_neighbors=n_neighbors)
        labels, scores = gather_scores(filter_model, docs)
        metrics = compute_threshold_metrics(labels, scores, threshold=filter_model.threshold)
        rows.append(
            {
                'ablation': 'neighbor_count',
                'setting': n_neighbors,
                'auc_proxy': 1.0 - (metrics['fnr_at_threshold'] + metrics['fpr_at_threshold']) / 2.0,
                'fnr': metrics['fnr_at_threshold'],
                'fpr': metrics['fpr_at_threshold'],
                'f1': metrics['f1_at_threshold'],
            }
        )

    for threshold in [1.3, 1.4, 1.5, 1.65, 1.8, 2.0]:
        filter_model = build_cqrcd_filter(retriever, smoke=args.smoke, threshold=threshold)
        labels, scores = gather_scores(filter_model, docs)
        metrics = compute_threshold_metrics(labels, scores, threshold=threshold)
        rows.append(
            {
                'ablation': 'threshold',
                'setting': threshold,
                'auc_proxy': 1.0 - (metrics['fnr_at_threshold'] + metrics['fpr_at_threshold']) / 2.0,
                'fnr': metrics['fnr_at_threshold'],
                'fpr': metrics['fpr_at_threshold'],
                'f1': metrics['f1_at_threshold'],
            }
        )

    for retriever_name in ['minilm', 'dpr']:
        try:
            current_retriever = init_retriever(retriever_name)
            filter_model = build_cqrcd_filter(current_retriever, smoke=args.smoke)
            labels, scores = gather_scores(filter_model, docs)
            metrics = compute_threshold_metrics(labels, scores, threshold=filter_model.threshold)
            rows.append(
                {
                    'ablation': 'retriever_backbone',
                    'setting': retriever_name,
                    'auc_proxy': 1.0 - (metrics['fnr_at_threshold'] + metrics['fpr_at_threshold']) / 2.0,
                    'fnr': metrics['fnr_at_threshold'],
                    'fpr': metrics['fpr_at_threshold'],
                    'f1': metrics['f1_at_threshold'],
                }
            )
        except Exception:
            rows.append(
                {
                    'ablation': 'retriever_backbone',
                    'setting': retriever_name,
                    'auc_proxy': None,
                    'fnr': None,
                    'fpr': None,
                    'f1': None,
                }
            )

    for method, smoke_mode in [('t5_or_fallback', False), ('fallback_only', True)]:
        filter_model = build_cqrcd_filter(retriever, smoke=smoke_mode)
        labels, scores = gather_scores(filter_model, docs)
        metrics = compute_threshold_metrics(labels, scores, threshold=filter_model.threshold)
        rows.append(
            {
                'ablation': 'neighbor_method',
                'setting': method,
                'auc_proxy': 1.0 - (metrics['fnr_at_threshold'] + metrics['fpr_at_threshold']) / 2.0,
                'fnr': metrics['fnr_at_threshold'],
                'fpr': metrics['fpr_at_threshold'],
                'f1': metrics['f1_at_threshold'],
            }
        )

    df = pd.DataFrame(rows)
    save_dataframe(df, outputs['tables'] / 'ablation_results.csv')

    neighbor_df = df[df['ablation'] == 'neighbor_count'].copy()
    neighbor_df['setting'] = neighbor_df['setting'].astype(float)
    threshold_df = df[df['ablation'] == 'threshold'].copy()
    threshold_df['setting'] = threshold_df['setting'].astype(float)

    save_figure(build_line_plot(neighbor_df, 'setting', 'auc_proxy', 'Ablation: neighbor count'), outputs['figures'] / 'fig5_ablation_n')
    threshold_df['fnr_plus_fpr'] = threshold_df['fnr'] + threshold_df['fpr']
    save_figure(build_line_plot(threshold_df, 'setting', 'fnr_plus_fpr', 'Ablation: threshold'), outputs['figures'] / 'fig6_ablation_threshold')

    print('Saved ablation outputs.')


if __name__ == '__main__':
    main()
