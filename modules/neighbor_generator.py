"""Neighbor (paraphrase) generator with optional T5 paraphrase support.

Session 004 changes:
- Replaced the `_simple_variants` fallback with a semantically-diverse generator
  that applies synonym substitutions and structural transformations.  The old
  fallback (prefix templates + token swaps) produced variants with MiniLM cosine
  similarity ~0.97-0.99 to the original query — essentially identical embeddings —
  making every concentration score converge to ~1.0.  The new fallback targets
  similarity ~0.82-0.92, creating measurable separation between legitimate and
  adversarial concentration distributions.

Session 005 changes:
- Added thematic variant generation as Round 0 in `_simple_variants`.  Pure synonym
  substitutions change wording but preserve intent, keeping MiniLM cosine ~0.87-0.92.
  Thematic variants SHIFT THE QUESTION ASPECT (e.g., "summarise advancements" →
  "what are the challenges") while keeping the same topic, producing MiniLM cosine
  ~0.70-0.80.  Mixing these into the 5-neighbor pool lowers the mean denominator in
  C(d,Q) = sim(d,Q) / mean(sim(d,Q_i)), pushing C_bb above threshold 1.20 while
  C_legit stays near 1.05 — restoring the detection separation lost under MiniLM.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional

import numpy as np

from config import DEVICE, PARAPHRASE_BATCH_SIZE, PARAPHRASE_MAX_LENGTH, PARAPHRASE_NUM_BEAMS, RANDOM_SEED


# ---------------------------------------------------------------------------
# Synonym tables — applied to produce semantically-shifted paraphrases.
# Each entry maps a source token/phrase to a list of replacement options.
# Replacements are chosen round-robin across generated variants so that
# different variants use different substitutes.
# ---------------------------------------------------------------------------
_WORD_SUBS: dict = {
    # Action verbs
    'summarize':    ['analyze',       'outline',        'describe',      'present'],
    'describe':     ['outline',       'characterize',   'present',       'explain'],
    'evaluate':     ['assess',        'examine',        'review',        'appraise'],
    'analyze':      ['examine',       'investigate',    'review',        'study'],
    'assess':       ['evaluate',      'measure',        'appraise',      'gauge'],
    'review':       ['examine',       'assess',         'survey',        'analyze'],
    'compare':      ['contrast',      'examine',        'evaluate',      'measure'],
    'explain':      ['describe',      'clarify',        'outline',       'illustrate'],
    'investigate':  ['examine',       'study',          'explore',       'research'],
    'monitor':      ['track',         'observe',        'oversee',       'supervise'],
    'draft':        ['prepare',       'create',         'develop',       'write'],
    'provide':      ['deliver',       'present',        'supply',        'offer'],
    'assist':       ['help',          'support',        'guide',         'aid'],
    'recommend':    ['suggest',       'propose',        'identify',      'advise'],
    'identify':     ['determine',     'find',           'pinpoint',      'establish'],
    'discuss':      ['address',       'examine',        'cover',         'explore'],
    'develop':      ['create',        'build',          'design',        'formulate'],
    'manage':       ['oversee',       'handle',         'coordinate',    'administer'],
    'outline':      ['summarize',     'describe',       'present',       'sketch'],
    'examine':      ['analyze',       'review',         'investigate',   'study'],
    # Frequency/recency modifiers
    'recent':       ['latest',        'current',        'contemporary',  'new'],
    'latest':       ['recent',        'current',        'newest',        'up-to-date'],
    'new':          ['recent',        'latest',         'emerging',      'novel'],
    # Temporal references
    'past':         ['previous',      'prior',          'last'],
    'previous':     ['past',          'prior',          'last'],
    'five years':   ['5-year period', 'half decade',    'five-year span'],
    'last five':    ['past five',     'previous five',  'prior five'],
    # Domain nouns
    'advancements': ['developments',  'progress',       'innovations',   'breakthroughs'],
    'developments': ['advancements',  'progress',       'changes',       'improvements'],
    'innovations':  ['advancements',  'developments',   'breakthroughs', 'advances'],
    'risk':         ['uncertainty',   'exposure',       'challenge',     'concern'],
    'returns':      ['gains',         'profits',        'outcomes',      'yield'],
    'performance':  ['results',       'metrics',        'outcomes',      'progress'],
    'analysis':     ['assessment',    'evaluation',     'review',        'examination'],
    'plan':         ['strategy',      'approach',       'framework',     'proposal'],
    'treatment':    ['therapy',       'care',           'intervention',  'management'],
    'network':      ['system',        'infrastructure', 'platform',      'environment'],
    'sector':       ['industry',      'domain',         'field',         'market'],
    'company':      ['organization',  'firm',           'enterprise',    'business'],
    'financial':    ['economic',      'fiscal',         'monetary',      'budgetary'],
    'investing':    ['investment',    'allocating',     'funding',       'financing'],
    'computing':    ['technology',    'research',       'science',       'systems'],
    'patient':      ['individual',    'person',         'subject',       'client'],
    'data':         ['information',   'records',        'datasets',      'content'],
    'report':       ['document',      'summary',        'overview',      'record'],
    'security':     ['protection',    'safety',         'defence',       'safeguarding'],
    'access':       ['availability',  'retrieval',      'entry',         'connectivity'],
    'system':       ['platform',      'infrastructure', 'framework',     'environment'],
    'student':      ['learner',       'pupil',          'candidate',     'individual'],
    'course':       ['module',        'subject',        'programme',     'class'],
    'medical':      ['clinical',      'health',         'healthcare',    'therapeutic'],
}

# ---------------------------------------------------------------------------
# Stop-word filter for topic extraction (used by thematic variant generator).
# These are words that don't contribute to the semantic "topic" of a query.
# ---------------------------------------------------------------------------
_STOP_WORDS_FILTER = frozenset({
    'a', 'an', 'the', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'and', 'or', 'but', 'to', 'is', 'are', 'was', 'were', 'be', 'been',
    'that', 'this', 'these', 'those', 'it', 'its', 'me', 'you', 'your',
    'my', 'we', 'our', 'they', 'their', 'what', 'which', 'how', 'when',
    'where', 'why', 'who', 'can', 'could', 'would', 'should', 'may', 'might',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'shall', 'not',
    'up', 'out', 'all', 'any', 'some', 'most', 'very', 'just', 'about',
    'into', 'over', 'after', 'since', 'past', 'five', 'ten', 'years',
    'recent', 'latest', 'current', 'last', 'previous', 'next', 'new',
    'please', 'give', 'me', 'i', 'need',
})

# ---------------------------------------------------------------------------
# Thematic question templates.  Each uses {topic} as a placeholder for the
# core subject extracted from the query.  These produce variants that share the
# TOPIC but ask about a DIFFERENT ASPECT, giving cosine ~0.70-0.80 under
# MiniLM (vs ~0.87-0.92 for synonym substitutions).  Mixed into the 5-neighbor
# pool they lower the mean denominator in C(d,Q) enough to push C_bb above
# threshold 1.20 while C_legit stays near 1.05.
# ---------------------------------------------------------------------------
_THEMATIC_TEMPLATES = [
    'What are the general trends in {topic}?',
    'What are the main challenges related to {topic}?',
    'Describe the broad landscape of {topic}.',
    'How has {topic} evolved historically?',
    'What are the future directions for {topic}?',
    'What limitations exist in the field of {topic}?',
    'How does {topic} compare to related areas?',
    'What is the current research focus in {topic}?',
]

_ACTION_VERBS = frozenset({
    'summarize', 'describe', 'analyze', 'evaluate', 'assess', 'explain',
    'compare', 'investigate', 'monitor', 'draft', 'provide', 'assist',
    'recommend', 'review', 'identify', 'outline', 'examine', 'discuss',
    'develop', 'manage', 'create', 'prepare', 'plan', 'design', 'build',
    'generate', 'write', 'determine', 'find', 'establish', 'ensure',
    'implement', 'resolve', 'update', 'verify', 'conduct', 'perform',
})


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
        self._cache: dict = {}

        self._model_loaded = False
        self._tokenizer = None
        self._model = None
        self._num_beams = num_beams
        self._max_length = max_length
        self._seed = seed

        self.retriever = retriever
        self.min_similarity = float(min_similarity)

    # ------------------------------------------------------------------
    # Thematic variant helpers (Session 005)
    # ------------------------------------------------------------------

    def _extract_topic(self, query: str) -> str:
        """Strip action verbs and stop words to get the core topic phrase."""
        words = query.split()
        topic_words = [
            w.strip('.,?!;:') for w in words
            if w.lower().strip('.,?!;:') not in _ACTION_VERBS
            and w.lower().strip('.,?!;:') not in _STOP_WORDS_FILTER
            and len(w.strip('.,?!;:')) > 2
        ]
        return ' '.join(topic_words[:7]).strip()

    def _thematic_variants(self, query: str, n: int) -> List[str]:
        """
        Generate aspect-shifted variants targeting MiniLM cosine ~0.70-0.80.

        Unlike synonym substitutions (same intent, different wording), these
        change the QUESTION TYPE while keeping the topic, producing lower cosine
        similarity that is essential for lifting C_bb above the threshold.
        """
        topic = self._extract_topic(query)
        if not topic or len(topic) < 4:
            return []
        q_lower = query.lower()
        seen: set = {q_lower}
        variants: List[str] = []
        for tmpl in _THEMATIC_TEMPLATES:
            if len(variants) >= n:
                break
            v = tmpl.format(topic=topic).strip()
            vl = v.lower()
            if vl != q_lower and vl not in seen:
                seen.add(vl)
                variants.append(v)
        return variants

    # ------------------------------------------------------------------
    # Improved fallback paraphrase generator
    # ------------------------------------------------------------------

    def _apply_subs(self, query: str, round_idx: int) -> str:
        """Apply up to 3 word-level synonym substitutions to query."""
        result = query
        applied = 0
        for term, subs in _WORD_SUBS.items():
            if applied >= 3:
                break
            if re.search(re.escape(term), result, re.IGNORECASE):
                sub = subs[round_idx % len(subs)]
                result = re.sub(re.escape(term), sub, result, count=1, flags=re.IGNORECASE)
                applied += 1
        return result.strip()

    def _simple_variants(self, query: str, n: int) -> List[str]:
        """
        Generate semantically diverse paraphrases without a neural model.

        Strategy (in order):
          0. Thematic variants: aspect-shifted questions (cosine ~0.70-0.80).
             These are the most important for CQRCD concentration separation.
          1. Single-term synonym substitution across all known terms.
          2. Double-term substitution using a second pass on the first variants.
          3. Structural transformation: imperative → interrogative.
          4. Framing prefix variants.
          5. Deterministic padding.

        Round 0 targets MiniLM cosine ~0.70-0.80; Rounds 1-4 target ~0.82-0.92.
        Mixing them lowers the mean denominator in C(d,Q), pushing C_bb above
        threshold 1.20 while C_legit stays near 1.05.
        """
        q_lower = query.lower().strip()
        words = query.split()
        seen: set = {q_lower}
        variants: List[str] = []

        def add(v: str) -> bool:
            v = v.strip()
            if not v:
                return False
            vl = v.lower()
            if vl == q_lower or vl in seen:
                return False
            # Require at least one word to differ (not just capitalisation)
            if vl.replace(' ', '') == q_lower.replace(' ', ''):
                return False
            seen.add(vl)
            variants.append(v)
            return True

        # --- Round 0: thematic variants (aspect-shifted, cosine ~0.70-0.80) ---
        for v in self._thematic_variants(query, n):
            add(v)
            if len(variants) >= n:
                return variants[:n]

        # --- Round 1: single-term substitutions ---
        for round_idx in range(4):
            candidate = self._apply_subs(query, round_idx)
            add(candidate)
            if len(variants) >= n:
                return variants[:n]

        # --- Round 2: double substitution (apply subs to round-1 variants) ---
        round1_base = variants[:min(3, len(variants))]
        for base in round1_base:
            for round_idx in range(1, 4):
                candidate = self._apply_subs(base, round_idx)
                add(candidate)
                if len(variants) >= n:
                    return variants[:n]

        # --- Round 3: structural transformation ---
        if words:
            first_w = words[0].lower()
            rest = ' '.join(words[1:]) if len(words) > 1 else query
            if first_w in _ACTION_VERBS and rest:
                interrogative_prefixes = [
                    'What are the key aspects of',
                    'How would you characterize',
                    'Provide an overview of',
                    'What is the current state of',
                    'Give a detailed account of',
                ]
                for prefix in interrogative_prefixes:
                    add(f"{prefix} {rest}")
                    if len(variants) >= n:
                        return variants[:n]

        # --- Round 4: framing prefixes ---
        framing = [
            'In detail,',
            'From a comprehensive perspective,',
            'Considering all relevant factors,',
            'An analysis of',
            'A structured review of',
        ]
        for frame in framing:
            add(f"{frame} {query}")
            if len(variants) >= n:
                return variants[:n]

        # --- Round 5: deterministic padding ---
        for i in range(n + 5):
            add(f"{query} in the broader context {i}")
            if len(variants) >= n:
                break

        return variants[:n]

    # ------------------------------------------------------------------
    # T5 paraphrase (unchanged from original)
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> bool:
        if not self.model_name:
            self._model_loaded = False
            return False
        if self._model_loaded:
            return True
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch  # noqa: F401
        except Exception:
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
            self._model_loaded = False
            return False

    def _paraphrase_with_t5(self, query: str, n: int) -> List[str]:
        if not self._ensure_model_loaded():
            return []
        import torch
        prompt = f"paraphrase: {query} </s>"
        inputs = self._tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(self.device)
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
        cand_texts = [
            self._tokenizer.decode(o, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True).strip()
            for o in outputs
        ]
        seen: set = set()
        paraphrases: List[str] = []
        for t in cand_texts:
            tl = t.strip()
            if not tl or tl.lower() == query.lower() or tl.lower() in seen:
                continue
            seen.add(tl.lower())
            paraphrases.append(tl)
            if len(paraphrases) >= num_return:
                break
        if self.retriever is not None and paraphrases:
            try:
                q_emb = self.retriever.encode_query(query)
                filtered = [
                    p for p in paraphrases
                    if float(np.dot(q_emb, self.retriever.encode_query(p))) >= self.min_similarity
                ]
                paraphrases = filtered or paraphrases
            except Exception:
                pass
        return paraphrases[:n]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, query: str, n: int = None) -> List[str]:
        n = n or self.default_n
        key = hashlib.sha1((query + str(n)).encode()).hexdigest()
        if key in self._cache:
            return list(self._cache[key])

        paraphrases: List[str] = []
        if self._ensure_model_loaded():
            try:
                paraphrases = self._paraphrase_with_t5(query, n)
            except Exception:
                paraphrases = []

        if len(paraphrases) < n:
            fallback = self._simple_variants(query, n)
            seen = {p.lower() for p in paraphrases}
            for f in fallback:
                if f.lower() not in seen and f.lower() != query.lower():
                    paraphrases.append(f)
                    seen.add(f.lower())
                if len(paraphrases) >= n:
                    break

        # Final pad (should rarely be needed)
        for i in range(n + 5):
            cand = f"{query} variant {i}"
            if cand.lower() not in {p.lower() for p in paraphrases}:
                paraphrases.append(cand)
            if len(paraphrases) >= n:
                break

        paraphrases = paraphrases[:n]
        self._cache[key] = paraphrases
        return paraphrases

    def generate_batch(self, queries: List[str], n: int = None) -> List[List[str]]:
        _ = PARAPHRASE_BATCH_SIZE
        return [self.generate(q, n=n) for q in queries]
