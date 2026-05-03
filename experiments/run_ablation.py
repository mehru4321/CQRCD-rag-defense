"""Experiment 5: ablation study for core CQRCD settings."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import compute_detection_metrics, compute_threshold_metrics
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
        text = doc.get('text', doc.get('full_text', ''))
        scores.append(filter_model.compute_concentration_score(text, query))
        labels.append(0 if doc.get('label') == 'legitimate' else 1)
    return labels, scores


def summarize_detection(labels, scores, threshold: float):
    detection = compute_detection_metrics(labels, scores)
    threshold_metrics = compute_threshold_metrics(labels, scores, threshold=threshold)
    return {
        'auc': detection['auc'],
        'optimal_threshold': detection['optimal_threshold'],
        'eval_threshold': threshold,
        'fnr_at_eval_threshold': threshold_metrics['fnr_at_threshold'],
        'fpr_at_eval_threshold': threshold_metrics['fpr_at_threshold'],
        'f1_at_eval_threshold': threshold_metrics['f1_at_threshold'],
        'tpr_at_eval_threshold': threshold_metrics['tpr_at_threshold'],
        'threshold_error': threshold_metrics['fnr_at_threshold'] + threshold_metrics['fpr_at_threshold'],
    }


def build_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df[x_col], df[y_col], marker='o')
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
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
        rows.append(
            {
                'ablation': 'neighbor_count',
                'setting': n_neighbors,
                **summarize_detection(labels, scores, threshold=filter_model.threshold),
            }
        )

    for threshold in [1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 2.0]:
        filter_model = build_cqrcd_filter(retriever, smoke=args.smoke, threshold=threshold)
        labels, scores = gather_scores(filter_model, docs)
        rows.append(
            {
                'ablation': 'threshold',
                'setting': threshold,
                **summarize_detection(labels, scores, threshold=threshold),
            }
        )

    for retriever_name in ['minilm', 'dpr']:
        try:
            current_retriever = init_retriever(retriever_name)
            filter_model = build_cqrcd_filter(current_retriever, smoke=args.smoke)
            labels, scores = gather_scores(filter_model, docs)
            rows.append(
                {
                    'ablation': 'retriever_backbone',
                    'setting': retriever_name,
                    **summarize_detection(labels, scores, threshold=filter_model.threshold),
                }
            )
        except Exception as exc:
            print(f"  WARNING: retriever '{retriever_name}' failed - {exc}")
            rows.append(
                {
                    'ablation': 'retriever_backbone',
                    'setting': retriever_name,
                    'auc': None,
                    'optimal_threshold': None,
                    'eval_threshold': None,
                    'fnr_at_eval_threshold': None,
                    'fpr_at_eval_threshold': None,
                    'f1_at_eval_threshold': None,
                    'tpr_at_eval_threshold': None,
                    'threshold_error': None,
                }
            )

    for method, smoke_mode in [('t5_or_fallback', False), ('fallback_only', True)]:
        filter_model = build_cqrcd_filter(retriever, smoke=smoke_mode)
        labels, scores = gather_scores(filter_model, docs)
        rows.append(
            {
                'ablation': 'neighbor_method',
                'setting': method,
                **summarize_detection(labels, scores, threshold=filter_model.threshold),
            }
        )

    df = pd.DataFrame(rows)
    save_dataframe(df, outputs['tables'] / 'ablation_results.csv')

    neighbor_df = df[df['ablation'] == 'neighbor_count'].copy()
    neighbor_df['setting'] = neighbor_df['setting'].astype(float)
    threshold_df = df[df['ablation'] == 'threshold'].copy()
    threshold_df['setting'] = threshold_df['setting'].astype(float)

    save_figure(
        build_line_plot(neighbor_df, 'setting', 'auc', 'Ablation: neighbor count', 'ROC-AUC'),
        outputs['figures'] / 'fig5_ablation_n',
    )
    save_figure(
        build_line_plot(threshold_df, 'setting', 'threshold_error', 'Ablation: threshold', 'FNR + FPR'),
        outputs['figures'] / 'fig6_ablation_threshold',
    )

    print('Saved ablation outputs.')


if __name__ == '__main__':
    main()
