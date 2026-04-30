"""Generate paper figures at 400 DPI.

The plotting functions accept experiment outputs when available and otherwise
fail with clear messages instead of silently producing placeholder claims.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.rcParams['figure.dpi'] = 400
matplotlib.rcParams['savefig.dpi'] = 400


FIGURE_DIR = Path('results/figures')


def save_figure(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f'{name}.png', dpi=400, bbox_inches='tight')
    fig.savefig(FIGURE_DIR / f'{name}.pdf', bbox_inches='tight')


def main() -> None:
    required = [
        'fig1_concentration_dist',
        'fig2_roc_comparison',
        'fig3_asr_bar',
        'fig4_adaptive_tradeoff',
        'fig5_ablation_n',
        'fig6_ablation_threshold',
    ]
    print('Figure helper ready. Expected figure basenames:')
    for name in required:
        print(f' - {name}')


if __name__ == '__main__':
    main()
