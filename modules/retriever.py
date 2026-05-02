"""Dense retriever wrapper supporting MiniLM and DPR.

This implements `DenseRetriever` with two backends:
- MiniLM via `sentence-transformers` (single encoder for queries/docs)
- DPR via HuggingFace transformers (separate question/context encoders)

All returned vectors are L2-normalized and are NumPy arrays of dtype float32.
"""
from typing import List, Optional
import numpy as np

from config import DEVICE, DPR_BATCH_SIZE, RETRIEVER_BATCH_SIZE


class DenseRetriever:
    def __init__(self, model_name: str = 'minilm', device: Optional[str] = None):
        self.model_name = model_name.lower()
        self.device = device or DEVICE
        # cache keyed by (role, text) where role in {'minilm','q','d'}
        self._cache = {}
        self._is_dpr = False

        if self.model_name == 'minilm':
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise ImportError('sentence-transformers is required for MiniLM: ' + str(e))
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
            self.dim = self.model.get_sentence_embedding_dimension()

        elif self.model_name == 'dpr' or self.model_name.startswith('facebook/dpr'):
            # DPR uses separate encoders for queries and contexts.
            # Must use the DPR-specific classes (not AutoModel) to get the
            # correct pooler_output projection layer in the forward pass.
            try:
                import torch
                from transformers import (
                    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
                    DPRContextEncoder, DPRContextEncoderTokenizer,
                )
            except Exception as e:
                raise ImportError('transformers and torch are required for DPR: ' + str(e))

            q_model_id = 'facebook/dpr-question_encoder-single-nq-base'
            d_model_id = 'facebook/dpr-ctx_encoder-single-nq-base'

            self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_model_id)
            self.q_model = DPRQuestionEncoder.from_pretrained(q_model_id)
            self.q_model.to(self.device)
            self.q_model.eval()

            self.d_tokenizer = DPRContextEncoderTokenizer.from_pretrained(d_model_id)
            self.d_model = DPRContextEncoder.from_pretrained(d_model_id)
            self.d_model.to(self.device)
            self.d_model.eval()

            # DPR hidden size is 768
            self.dim = self.q_model.config.projection_dim or self.q_model.config.hidden_size
            self._is_dpr = True

        else:
            raise NotImplementedError('Supported retrievers: minilm, dpr')

    def _normalize(self, emb: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(emb, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        return emb / norms

    def _mean_pool(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, dim), attention_mask: (batch, seq_len)
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        counts = counts.clamp(min=1e-8)
        return summed / counts

    def encode_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Batch-encode texts. For DPR this uses the question encoder (queries).

        Returns: (N, dim) float32 NumPy array, L2-normalized.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if not self._is_dpr:
            emb = self.model.encode(
                texts,
                batch_size=batch_size or RETRIEVER_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            emb = np.asarray(emb, dtype=np.float32)
            emb = self._normalize(emb)
            return emb

        # DPR: use question encoder for batch encoding (suitable for queries/neighbors).
        # DPR canonical embedding is pooler_output (CLS → linear projection), not
        # mean pooling over last_hidden_state.
        import torch
        batch_size = batch_size or DPR_BATCH_SIZE
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.q_tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors='pt')
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.q_model(**enc)
                chunks.append(out.pooler_output.cpu().numpy().astype('float32'))
        return self._normalize(np.vstack(chunks).astype('float32'))

    def encode_query(self, query: str) -> np.ndarray:
        key = ('q' if self._is_dpr else 'minilm', query)
        if key in self._cache:
            return self._cache[key]

        if not self._is_dpr:
            emb = self.encode_batch([query])[0]
            self._cache[key] = emb
            return emb

        # DPR query encoding — use pooler_output (canonical DPR embedding).
        import torch
        enc = self.q_tokenizer(query, padding=True, truncation=True,
                               max_length=512, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.q_model(**enc)
            emb = out.pooler_output.cpu().numpy().astype('float32')[0]
            emb = self._normalize(emb)
            self._cache[key] = emb
            return emb

    def encode_document(self, document: str) -> np.ndarray:
        key = ('d' if self._is_dpr else 'minilm', document)
        if key in self._cache:
            return self._cache[key]

        if not self._is_dpr:
            emb = self.encode_batch([document])[0]
            self._cache[key] = emb
            return emb

        # DPR document/context encoding — use pooler_output (canonical DPR embedding).
        import torch
        enc = self.d_tokenizer(document, padding=True, truncation=True,
                               max_length=512, return_tensors='pt')
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.d_model(**enc)
            emb = out.pooler_output.cpu().numpy().astype('float32')[0]
            emb = self._normalize(emb)
            self._cache[key] = emb
            return emb

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        # Both vectors expected to be L2-normalized
        return float(np.dot(vec_a, vec_b))
