"""FAISS-backed knowledge base manager."""
from typing import List, Dict, Optional
import numpy as np

from config import FAISS_GPU_MEMORY_MB, FAISS_USE_GPU


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = FAISS_USE_GPU):
    """Build an inner-product FAISS index with optional GPU acceleration."""
    import faiss

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    if embeddings.size:
        index.add(embeddings.astype('float32'))

    if not use_gpu:
        return index, None

    try:
        if not hasattr(faiss, 'get_num_gpus') or faiss.get_num_gpus() <= 0:
            return index, None
        resources = faiss.StandardGpuResources()
        resources.setTempMemory(FAISS_GPU_MEMORY_MB * 1024 * 1024)
        return faiss.index_cpu_to_gpu(resources, 0, index), resources
    except Exception:
        return index, None


class KnowledgeBase:
    def __init__(self, retriever, use_gpu: bool = FAISS_USE_GPU):
        self.retriever = retriever
        self.dim = retriever.dim
        try:
            import faiss
        except Exception as e:
            raise ImportError('faiss is required for KnowledgeBase: ' + str(e))
        self.faiss = faiss
        self.use_gpu = use_gpu
        self._gpu_resources = None
        self.index = self._new_index()
        self.docs: List[Dict] = []
        self.id_to_idx = {}

    def _new_index(self):
        empty = np.zeros((0, self.dim), dtype='float32')
        index, resources = build_faiss_index(empty, use_gpu=self.use_gpu)
        self._gpu_resources = resources
        return index

    def add_documents(self, documents: List[Dict]) -> None:
        """Add a list of documents to the FAISS index.

        Each document: {'id': str, 'text': str, 'label': str, ...}
        """
        if not documents:
            return

        embeddings = []
        for doc in documents:
            if 'id' not in doc or ('text' not in doc and 'full_text' not in doc):
                raise ValueError('Each document must include "id" and either "text" or "full_text"')
            doc = dict(doc)
            doc.setdefault('text', doc.get('full_text', ''))
            idx = len(self.docs)
            self.id_to_idx[doc['id']] = idx
            self.docs.append(doc)
            emb = self.retriever.encode_document(doc['text']).astype('float32')
            embeddings.append(emb)

        xb = np.ascontiguousarray(np.vstack(embeddings).astype('float32'))
        self.index.add(xb)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        q_emb = self.retriever.encode_query(query).astype('float32').reshape(1, -1)
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = dict(self.docs[idx])
            doc['score'] = float(score)
            doc['index'] = int(idx)
            results.append(doc)
        return results

    def poison(self, adversarial_docs: List[Dict]) -> None:
        """Add adversarial documents to simulate poisoning."""
        self.add_documents(adversarial_docs)

    def reset(self) -> None:
        """Clear the index and documents."""
        self.index = self._new_index()
        self.docs = []
        self.id_to_idx = {}

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        idx = self.id_to_idx.get(doc_id)
        if idx is None:
            return None
        return self.docs[idx]
