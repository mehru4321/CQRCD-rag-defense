"""CQRCD filter: compute concentration scores and filter retrieved docs."""
from typing import List, Dict, Tuple, Union
import numpy as np


class CQRCDFilter:
    def __init__(
        self,
        retriever,
        neighbor_generator,
        threshold: float = 1.65,
        n_neighbors: int = 5,
        epsilon: float = 1e-8
    ):
        self.retriever = retriever
        self.neighbor_generator = neighbor_generator
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

    def compute_concentration_score(
        self,
        document_text: str,
        query: str,
        neighbors: List[str] = None,
        neighbor_embs: np.ndarray = None,
        query_emb: np.ndarray = None,
        doc_emb: np.ndarray = None,
    ) -> float:
        if neighbors is None:
            neighbors = self.neighbor_generator.generate(query, n=self.n_neighbors)

        doc_emb = doc_emb if doc_emb is not None else self.retriever.encode_document(document_text)
        q_emb = query_emb if query_emb is not None else self.retriever.encode_query(query)
        sim_q = self.retriever.similarity(doc_emb, q_emb)

        if neighbor_embs is None:
            neighbor_embs = self.retriever.encode_batch(neighbors)
        sims = [self.retriever.similarity(doc_emb, nb) for nb in neighbor_embs]
        mean_neighbor_sim = float(np.mean(sims)) if len(sims) > 0 else 0.0
        score = sim_q / max(mean_neighbor_sim, self.epsilon)
        return float(score)

    def score_all(self, query: str, documents: List[Dict]) -> List[Tuple[Dict, float]]:
        neighbors = self.neighbor_generator.generate(query, n=self.n_neighbors)
        neighbor_embs = self.retriever.encode_batch(neighbors)
        query_emb = self.retriever.encode_query(query)
        results = []
        for doc in documents:
            text = doc.get('text', doc.get('full_text', ''))
            score = self.compute_concentration_score(
                text,
                query,
                neighbors,
                neighbor_embs=neighbor_embs,
                query_emb=query_emb,
            )
            results.append((doc, score))
        return results

    def filter(self, query: str, retrieved_docs: List[Dict], return_scores: bool = False) -> Union[List[Dict], Tuple[List[Dict], List[float]]]:
        neighbors = self.neighbor_generator.generate(query, n=self.n_neighbors)
        neighbor_embs = self.retriever.encode_batch(neighbors)
        query_emb = self.retriever.encode_query(query)
        scores = []
        kept = []
        for doc in retrieved_docs:
            text = doc.get('text', doc.get('full_text', ''))
            score = self.compute_concentration_score(
                text,
                query,
                neighbors,
                neighbor_embs=neighbor_embs,
                query_emb=query_emb,
            )
            scores.append(score)
            if score <= self.threshold:
                kept.append(doc)

        if return_scores:
            return kept, scores
        return kept

    def find_optimal_threshold(self, validation_data: List[Dict]) -> float:
        # Basic grid search over plausible thresholds
        y_true = []
        y_scores = []
        for item in validation_data:
            query = item['query']
            doc = item['document']
            label = item.get('label', doc.get('label', 'legitimate'))
            score = self.compute_concentration_score(doc.get('text', doc.get('full_text', '')), query)
            y_scores.append(score)
            y_true.append(1 if label == 'adversarial' else 0)

        # compute Youden's J statistic-like minimization of FPR+FNR
        best_t = self.threshold
        best_val = float('inf')
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        thresholds = np.linspace(0.5, 3.5, 301)
        for t in thresholds:
            preds = (y_scores > t).astype(int)
            fn = ((y_true == 1) & (preds == 0)).sum()
            fp = ((y_true == 0) & (preds == 1)).sum()
            fnr = fn / max(1, (y_true == 1).sum())
            fpr = fp / max(1, (y_true == 0).sum())
            val = fnr + fpr
            if val < best_val:
                best_val = val
                best_t = float(t)
        return best_t
