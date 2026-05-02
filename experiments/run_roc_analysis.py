"""Experiment 2: ROC analysis for CQRCD and baselines."""
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
from modules.baseline_defenses import LLMBasedDefense, PerplexityDefense


def compute_cqrcd_scores(filter_model, docs):
    rows = []
    for doc in docs:
        query = doc.get('target_query') or doc.get('source_query') or doc.get('text', '')
        score = filter_model.compute_concentration_score(doc.get('text', doc.get('full_text', '')), query)
        rows.append({'doc_id': doc['id'], 'score': score, 'query': query})
    return rows


def build_roc_plot(curves):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, curve in curves.items():
        fpr, tpr = curve
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_title('ROC comparison: CQRCD vs baselines')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Experiment 2 ROC analysis.')
    parser.add_argument('--retriever', default='minilm', choices=['minilm', 'dpr'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--use-real-llm', action='store_true',
                        help='Load Qwen2-7B-Instruct for the LLM baseline (requires ~4.2 GB '
                             'VRAM on top of the retriever — do NOT combine with --retriever dpr). '
                             'Defaults to smoke (keyword) mode to stay within the 8 GB VRAM budget.')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    outputs = ensure_output_dirs(args.output_dir)
    bundle = load_dataset_bundle()
    legitimate = bundle['legitimate']
    adversarial = bundle['blackbox'] + bundle['whitebox']
    if args.smoke:
        legitimate = sample_for_smoke(legitimate, limit=20)
        adversarial = sample_for_smoke(adversarial, limit=60)

    docs = legitimate + adversarial
    y_true = [0] * len(legitimate) + [1] * len(adversarial)

    retriever = init_retriever(args.retriever)
    cqrcd = build_cqrcd_filter(retriever, smoke=args.smoke)
    cqrcd_rows = compute_cqrcd_scores(cqrcd, docs)
    cqrcd_scores = [row['score'] for row in cqrcd_rows]
    cqrcd_metrics = compute_detection_metrics(y_true, cqrcd_scores)
    cqrcd_threshold_metrics = compute_threshold_metrics(y_true, cqrcd_scores, cqrcd.threshold)

    ppl = PerplexityDefense()
    ppl_scores = ppl.score_documents(docs)
    ppl_metrics = compute_detection_metrics(y_true, ppl_scores)
    ppl_threshold = ppl.default_threshold(ppl_scores)
    ppl_threshold_metrics = compute_threshold_metrics(y_true, ppl_scores, ppl_threshold)

    # Exp 2 (ROC analysis) does not need a real LLM — only FNR/FPR point estimates.
    # The RTX 4070 guide says "Experiments 1, 2, 4, 5: no LLM needed".
    # Use --use-real-llm only when you specifically want the real LLM baseline.
    llm_mode = 'real' if (args.use_real_llm and not args.smoke) else 'smoke'
    llm = LLMBasedDefense(mode=llm_mode)
    llm_scores = []
    for doc in docs:
        query = doc.get('target_query') or doc.get('source_query') or doc.get('text', '')
        llm_scores.append(llm.score_documents(query, [doc])[0])
    llm_metrics = compute_threshold_metrics(y_true, llm_scores, threshold=0.5)

    rows = [
        {
            'method': 'cqrcd',
            'auc': cqrcd_metrics['auc'],
            'threshold': cqrcd.threshold,
            'fnr_at_threshold': cqrcd_threshold_metrics['fnr_at_threshold'],
            'fpr_at_threshold': cqrcd_threshold_metrics['fpr_at_threshold'],
            'f1_at_threshold': cqrcd_threshold_metrics['f1_at_threshold'],
        },
        {
            'method': 'ppl',
            'auc': ppl_metrics['auc'],
            'threshold': ppl_threshold,
            'fnr_at_threshold': ppl_threshold_metrics['fnr_at_threshold'],
            'fpr_at_threshold': ppl_threshold_metrics['fpr_at_threshold'],
            'f1_at_threshold': ppl_threshold_metrics['f1_at_threshold'],
        },
        {
            'method': f'llm_{llm.mode}',
            'auc': None,
            'threshold': 0.5,
            'fnr_at_threshold': llm_metrics['fnr_at_threshold'],
            'fpr_at_threshold': llm_metrics['fpr_at_threshold'],
            'f1_at_threshold': llm_metrics['f1_at_threshold'],
        },
    ]
    metrics_df = pd.DataFrame(rows)
    save_dataframe(metrics_df, outputs['tables'] / 'detection_metrics.csv')

    curves = {
        f"CQRCD (AUC={cqrcd_metrics['auc']:.3f})": cqrcd_metrics['roc_curve'][:2],
        f"PPL (AUC={ppl_metrics['auc']:.3f})": ppl_metrics['roc_curve'][:2],
    }
    save_figure(build_roc_plot(curves), outputs['figures'] / 'fig2_roc_comparison')
    save_dataframe(pd.DataFrame(cqrcd_rows), outputs['tables'] / 'cqrcd_detection_scores.csv')
    roc_rows = []
    for method, metrics in [('cqrcd', cqrcd_metrics), ('ppl', ppl_metrics)]:
        fpr, tpr, thresholds = metrics['roc_curve']
        for fp_rate, tp_rate, threshold in zip(fpr, tpr, thresholds):
            roc_rows.append(
                {
                    'method': method,
                    'fpr': float(fp_rate),
                    'tpr': float(tp_rate),
                    'threshold': float(threshold),
                }
            )
    save_dataframe(pd.DataFrame(roc_rows), outputs['tables'] / 'roc_curve_points.csv')

    print('Saved detection metrics and ROC comparison.')


if __name__ == '__main__':
    main()
