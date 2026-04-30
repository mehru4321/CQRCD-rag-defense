"""Baseline defenses: Perplexity-based and LLM-based stubs.

These are lightweight implementations suitable for initial evaluation. The
Perplexity defense uses GPT-2 via transformers to compute pseudo-perplexity.
LLM-based defense is a stub that can be connected to a real LLM later.
"""
from typing import List, Dict
import math

from config import DEVICE, MAX_TOKENS


class PerplexityDefense:
    def __init__(self, model_name: str = 'gpt2'):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception:
            raise ImportError('transformers is required for PerplexityDefense')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(DEVICE)
        self.model.eval()

    def compute_perplexity(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_TOKENS)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        input_ids = enc['input_ids']
        with __import__('torch').no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs[0].item()
        return math.exp(loss)

    def rerank(self, documents: List[Dict]) -> List[Dict]:
        docs = list(documents)
        scored = []
        for d in docs:
            ppl = self.compute_perplexity(d.get('text', d.get('full_text', '')))
            d = dict(d)
            d['_ppl'] = ppl
            scored.append(d)
        scored.sort(key=lambda x: x['_ppl'])
        return scored

    def filter(self, documents: List[Dict], threshold: float = None) -> List[Dict]:
        if threshold is None:
            return documents
        return [d for d in documents if self.compute_perplexity(d.get('text', d.get('full_text', ''))) <= threshold]


class LLMBasedDefense:
    def __init__(self, llm_backbone=None):
        self.llm = llm_backbone

    def classify(self, document: str, query: str) -> str:
        # Placeholder: always returns 'legitimate' (high FNR can be simulated in evaluation)
        return 'legitimate'

    def filter(self, query: str, documents: List[Dict]) -> List[Dict]:
        return [d for d in documents if self.classify(d.get('text', d.get('full_text', '')), query) == 'legitimate']
