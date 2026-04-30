"""Shared evaluation pipeline helpers."""
from __future__ import annotations

from typing import Dict, List

from evaluation.metrics import compute_asr_a, compute_asr_r, compute_retrieval_rate


class Evaluator:
    """Run an agent over a dataset and compute aggregate ASR/RR metrics."""

    def __init__(self, agent):
        self.agent = agent

    def evaluate_agent(self, dataset: List[Dict], use_defense: bool = True) -> Dict:
        results = []
        original_defense = self.agent.defense
        if not use_defense:
            self.agent.defense = None
        try:
            for item in dataset:
                results.append(
                    self.agent.run_task(
                        item['query'],
                        item.get('available_tools', []),
                        attack_tool=item.get('attack_tool'),
                    )
                )
        finally:
            self.agent.defense = original_defense

        return {
            'results_per_task': results,
            'aggregate': {
                'ASR_A': compute_asr_a(results),
                'ASR_R': compute_asr_r(results),
                'RR': compute_retrieval_rate(results),
                'defense_active': use_defense,
            },
        }
