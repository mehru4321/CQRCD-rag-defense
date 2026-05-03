"""Experiment 5: ablation study for core CQRCD settings."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from config import get_retriever_threshold
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
from modules.neighbor_generator import NeighborGenerator


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


def build_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    valid = df.dropna(subset=[y_col])
    bars = ax.bar(valid[x_col].astype(str), valid[y_col])
    # Label each bar with its value
    for bar, val in zip(bars, valid[y_col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.05)
    return fig


def threshold_sweep(retriever_name: str):
    if retriever_name == 'dpr':
        return [0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    return [1.2, 1.3, 1.4, 1.5, 1.65, 1.8, 2.0]


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
    eval_threshold = get_retriever_threshold(args.retriever)
    rows = []

    for n_neighbors in [1, 3, 5, 10, 20]:
        filter_model = build_cqrcd_filter(
            retriever,
            smoke=args.smoke,
            n_neighbors=n_neighbors,
            threshold=eval_threshold,
        )
        labels, scores = gather_scores(filter_model, docs)
        rows.append(
            {
                'ablation': 'neighbor_count',
                'setting': n_neighbors,
                **summarize_detection(labels, scores, threshold=eval_threshold),
            }
        )

    for threshold in threshold_sweep(args.retriever):
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
            current_threshold = get_retriever_threshold(retriever_name)
            filter_model = build_cqrcd_filter(current_retriever, smoke=args.smoke, threshold=current_threshold)
            labels, scores = gather_scores(filter_model, docs)
            rows.append(
                {
                    'ablation': 'retriever_backbone',
                    'setting': retriever_name,
                    **summarize_detection(labels, scores, threshold=current_threshold),
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

    # --- Neighbor method ablation ---
    # Check T5 availability once upfront so we can report it clearly.
    print("  Checking T5 paraphrase model availability...")
    _t5_probe = NeighborGenerator(model_name='Vamsi/T5_Paraphrase_Paws', retriever=retriever)
    t5_available = _t5_probe._ensure_model_loaded()
    print(f"  T5 available: {t5_available}")

    # Three guaranteed-distinct fallback conditions.
    for variant_mode in ['mixed', 'thematic_only', 'synonym_only']:
        filter_model = build_cqrcd_filter(
            retriever, smoke=False, variant_mode=variant_mode,
            threshold=eval_threshold,
        )
        labels, scores = gather_scores(filter_model, docs)
        rows.append(
            {
                'ablation': 'neighbor_method',
                'setting': variant_mode,
                **summarize_detection(labels, scores, threshold=eval_threshold),
            }
        )

    # T5 condition: neural paraphrase primary + mixed fallback.
    if t5_available:
        filter_model = build_cqrcd_filter(retriever, smoke=False, variant_mode='mixed')
        labels, scores = gather_scores(filter_model, docs)
    else:
        labels, scores = [], []
    rows.append(
        {
            'ablation': 'neighbor_method',
            'setting': 't5_mixed',
            **(
                summarize_detection(labels, scores, threshold=eval_threshold)
                if t5_available
                else {k: None for k in summarize_detection([0, 1], [0.0, 2.0], threshold=eval_threshold)}
            ),
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

    method_df = df[df['ablation'] == 'neighbor_method'].copy()
    method_labels = {'mixed': 'Mixed\n(thematic+syn)', 'thematic_only': 'Thematic\nonly',
                     'synonym_only': 'Synonym\nonly', 't5_mixed': 'T5+Mixed'}
    method_df['setting'] = method_df['setting'].map(lambda x: method_labels.get(x, x))
    save_figure(
        build_bar_plot(method_df, 'setting', 'auc', 'Ablation: neighbor generation method', 'ROC-AUC'),
        outputs['figures'] / 'fig7_ablation_method',
    )

    print('Saved ablation outputs.')


if __name__ == '__main__':
    main()
