"""Regenerate all paper figures from saved result tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import save_figure

matplotlib.rcParams['figure.dpi'] = 400
matplotlib.rcParams['savefig.dpi'] = 400


DEFAULT_RESULTS_DIR = Path('results')
TABLE_DIR = DEFAULT_RESULTS_DIR / 'tables'
FIGURE_DIR = DEFAULT_RESULTS_DIR / 'figures'


def require_table(table_dir: Path, name: str) -> pd.DataFrame:
    path = table_dir / name
    if not path.exists():
        raise FileNotFoundError(f'Missing required table: {path}')
    return pd.read_csv(path)


def plot_concentration(table_dir: Path, figure_dir: Path):
    df = require_table(table_dir, 'concentration_scores.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for group in ['legitimate', 'blackbox', 'whitebox']:
        subset = df[df['group'] == group]['score']
        ax.hist(subset, bins=20, alpha=0.5, label=group, density=True)
    ax.set_title('CQRCD concentration score distributions')
    ax.set_xlabel('Concentration score')
    ax.set_ylabel('Density')
    ax.legend()
    save_figure(fig, figure_dir / 'fig1_concentration_dist')


def plot_roc(table_dir: Path, figure_dir: Path):
    points = require_table(table_dir, 'roc_curve_points.csv')
    metrics = require_table(table_dir, 'detection_metrics.csv')
    auc_lookup = {
        row['method']: row['auc']
        for _, row in metrics.iterrows()
        if pd.notna(row['auc'])
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in sorted(points['method'].unique()):
        subset = points[points['method'] == method]
        auc = auc_lookup.get(method)
        label = f"{method.upper()} (AUC={auc:.3f})" if auc is not None and pd.notna(auc) else method.upper()
        ax.plot(subset['fpr'], subset['tpr'], label=label)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    ax.set_title('ROC comparison: CQRCD vs baselines')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend()
    save_figure(fig, figure_dir / 'fig2_roc_comparison')


def plot_asr(table_dir: Path, figure_dir: Path):
    df = require_table(table_dir, 'asr_results.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df['defense'], df['ASR_A'], color=['#7f8c8d', '#e67e22', '#3498db', '#2ecc71'][:len(df)])
    ax.set_title('Attack success under defense conditions')
    ax.set_ylabel('ASR_A')
    save_figure(fig, figure_dir / 'fig3_asr_bar')


def plot_adaptive(table_dir: Path, figure_dir: Path):
    df = require_table(table_dir, 'adaptive_tradeoff.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['c_target'], df['ASR_A'], marker='o', label='ASR_A')
    ax.plot(df['c_target'], df['detection_rate'], marker='s', label='Detection rate')
    if 'mean_concentration' in df.columns:
        ax.plot(df['c_target'], df['mean_concentration'], marker='^', label='Mean concentration')
    ax.set_title('Adaptive attacker tradeoff')
    ax.set_xlabel('Target concentration')
    ax.set_ylabel('Value')
    ax.legend()
    save_figure(fig, figure_dir / 'fig4_adaptive_tradeoff')


def plot_ablation(table_dir: Path, figure_dir: Path):
    df = require_table(table_dir, 'ablation_results.csv')

    neighbor_df = df[df['ablation'] == 'neighbor_count'].copy()
    neighbor_df['setting'] = neighbor_df['setting'].astype(float)
    fig1, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(neighbor_df['setting'], neighbor_df['auc'], marker='o')
    ax1.set_title('Ablation: neighbor count')
    ax1.set_xlabel('Neighbors')
    ax1.set_ylabel('ROC-AUC')
    save_figure(fig1, figure_dir / 'fig5_ablation_n')

    threshold_df = df[df['ablation'] == 'threshold'].copy()
    threshold_df['setting'] = threshold_df['setting'].astype(float)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    ax2.plot(threshold_df['setting'], threshold_df['threshold_error'], marker='o')
    ax2.set_title('Ablation: threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('FNR + FPR')
    save_figure(fig2, figure_dir / 'fig6_ablation_threshold')

    method_df = df[df['ablation'] == 'neighbor_method'].copy()
    if not method_df.empty:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.bar(method_df['setting'], method_df['auc'])
        ax3.set_title('Ablation: neighbor generation method')
        ax3.set_xlabel('Method')
        ax3.set_ylabel('ROC-AUC')
        save_figure(fig3, figure_dir / 'fig7_ablation_method')


def main() -> None:
    parser = argparse.ArgumentParser(description='Regenerate all paper figures from saved result tables.')
    parser.add_argument('--input-dir', default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument('--output-dir', default=None,
                        help='Optional output directory for figures. Defaults to <input-dir>/figures.')
    args = parser.parse_args()

    results_dir = Path(args.input_dir)
    table_dir = results_dir / 'tables'
    figure_dir = Path(args.output_dir) / 'figures' if args.output_dir else results_dir / 'figures'

    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_concentration(table_dir, figure_dir)
    plot_roc(table_dir, figure_dir)
    plot_asr(table_dir, figure_dir)
    plot_adaptive(table_dir, figure_dir)
    plot_ablation(table_dir, figure_dir)
    print('Regenerated all figures from saved tables.')


if __name__ == '__main__':
    main()
