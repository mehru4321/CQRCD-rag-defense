"""Neighbor (paraphrase) generator with optional T5 paraphrase support.

Behavior:
- Lazy-loads a T5 paraphrase model (`Vamsi/T5_Paraphrase_Paws` by default) when
  `generate()` is called. If the model or `transformers` are unavailable, falls
  back to a deterministic lexical variant generator.
- Optionally accepts a `retriever` to filter paraphrases by semantic similarity
  (cosine) to the original query (threshold `min_similarity`). This is useful
  to discard low-quality paraphrases.
"""
from typing import List, Optional
import hashlib
import numpy as np

from config import DEVICE, PARAPHRASE_BATCH_SIZE, PARAPHRASE_MAX_LENGTH, PARAPHRASE_NUM_BEAMS, RANDOM_SEED


class NeighborGenerator:
    def __init__(
        self,
        model_name: str = 'Vamsi/T5_Paraphrase_Paws',
        n: int = 5,
        device: str = DEVICE,
        retriever: Optional[object] = None,
        min_similarity: float = 0.5,
        num_beams: int = PARAPHRASE_NUM_BEAMS,
        max_length: int = PARAPHRASE_MAX_LENGTH,
        seed: int = RANDOM_SEED,
    ):
        self.model_name = model_name
        self.default_n = n
        self.device = device
        self._cache = {}

        # Lazy-loaded model attributes
        self._model_loaded = False
        self._tokenizer = None
        self._model = None
        self._num_beams = num_beams
        self._max_length = max_length
        self._seed = seed

        # Optional retriever for semantic filtering of paraphrases
        self.retriever = retriever
        self.min_similarity = float(min_similarity)

    def _simple_variants(self, query: str, n: int) -> List[str]:
        words = query.split()
        variants = []
        # small deterministic templates
        templates = [
            '{}',
            'Please {}',
            'Can you {}',
            '{} please',
            'Provide a summary: {}',
            'How to: {}'
        ]
        base = ' '.join(words[:7]) if words else query
        for t in templates:
            s = t.format(base).strip()
            if s.lower() != query.lower() and s not in variants:
                variants.append(s)
            if len(variants) >= n:
                break

        # token swaps
        for i in range(min(3, max(0, len(words) - 1))):
            w2 = words.copy()
            w2[i], w2[i+1] = w2[i+1], w2[i]
            s = ' '.join(w2).strip()
            if s.lower() != query.lower() and s not in variants:
                variants.append(s)
            if len(variants) >= n:
                break

        # truncation/expansion
        if len(variants) < n:
            trunc = ' '.join(words[: max(1, len(words)-1)])
            if trunc and trunc.lower() != query.lower() and trunc not in variants:
                variants.append(trunc)
        if len(variants) < n:
            variants.append((query + ' please').strip())

        # pad deterministically if still short
        i = 0
        while len(variants) < n:
            cand = f"{query} variant {i}"
            if cand.lower() != query.lower() and cand not in variants:
                variants.append(cand)
            i += 1

        return variants[:n]

    def _ensure_model_loaded(self) -> bool:
        if not self.model_name:
            self._model_loaded = False
            return False
        if self._model_loaded:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
        except Exception:
            # transformers not available
            self._model_loaded = False
            return False

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            self._model_loaded = True
            return True
        except Exception:
            # model download/load failed; fall back
            self._model_loaded = False
            return False

    def _paraphrase_with_t5(self, query: str, n: int) -> List[str]:
        if not self._ensure_model_loaded():
            return []

        from transformers import AutoTokenizer
        import torch

        # Prepare input prompt expected by paraphrase T5 models
        prompt = f"paraphrase: {query} </s>"
        inputs = self._tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(self.device)

        # choose number of returned candidates (generate a few extra to allow filtering)
        requested = max(1, n)
        num_return = min(self._num_beams, max(requested * 2, requested))

        gen = torch.Generator(device=self.device)
        gen.manual_seed(self._seed)

        try:
            outputs = self._model.generate(
                inputs,
                num_beams=self._num_beams,
                num_return_sequences=num_return,
                max_length=self._max_length,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False,
                generator=gen,
            )
        except Exception:
            return []

        cand_texts = [self._tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for out in outputs]

        # Deduplicate, exclude original, preserve order
        seen = set()
        paraphrases = []
        for t in cand_texts:
            tl = t.strip()
            if not tl:
                continue
            key = tl.lower()
            if key == query.lower():
                continue
            if key in seen:
                continue
            seen.add(key)
            paraphrases.append(tl)
            if len(paraphrases) >= num_return:
                break

        # Optional semantic filtering using retriever embeddings
        if self.retriever is not None and len(paraphrases) > 0:
            try:
                q_emb = self.retriever.encode_query(query)
                filtered = []
                for p in paraphrases:
                    p_emb = self.retriever.encode_query(p)
                    sim = float(np.dot(q_emb, p_emb))
                    if sim >= self.min_similarity:
                        filtered.append(p)
                paraphrases = filtered
            except Exception:
                # if retriever fails for any reason, ignore filtering
                pass

        return paraphrases[:n]

    def generate(self, query: str, n: int = None) -> List[str]:
        n = n or self.default_n
        key = hashlib.sha1((query + str(n)).encode()).hexdigest()
        if key in self._cache:
            return list(self._cache[key])

        # Try T5 paraphrase first (lazy-load). If unavailable or yields too few
        # high-quality paraphrases, fall back to deterministic generator.
        paraphrases = []
        if self._ensure_model_loaded():
            try:
                paraphrases = self._paraphrase_with_t5(query, n)
            except Exception:
                paraphrases = []

        # If model not available or produced too few candidates, use fallback
        if len(paraphrases) < n:
            fallback = self._simple_variants(query, n)
            # merge while preserving uniqueness
            seen = set([p.lower() for p in paraphrases])
            for f in fallback:
                if f.lower() not in seen and f.lower() != query.lower():
                    paraphrases.append(f)
                    seen.add(f.lower())
                if len(paraphrases) >= n:
                    break

        # Final padding if necessary
        i = 0
        while len(paraphrases) < n:
            cand = f"{query} variant {i}"
            if cand.lower() != query.lower() and cand.lower() not in {p.lower() for p in paraphrases}:
                paraphrases.append(cand)
            i += 1

        paraphrases = paraphrases[:n]
        self._cache[key] = paraphrases
        return paraphrases

    def generate_batch(self, queries: List[str], n: int = None) -> List[List[str]]:
        _ = PARAPHRASE_BATCH_SIZE
        return [self.generate(q, n=n) for q in queries]

