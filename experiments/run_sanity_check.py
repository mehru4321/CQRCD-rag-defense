"""Sanity check script: encode small set of docs and compute concentration scores."""
from modules.retriever import DenseRetriever
from modules.knowledge_base import KnowledgeBase
from modules.dsrm_simulator import DSRMSimulator
from modules.neighbor_generator import NeighborGenerator
from modules.cqrcd_filter import CQRCDFilter
import matplotlib
matplotlib.rcParams['figure.dpi'] = 400
matplotlib.rcParams['savefig.dpi'] = 400
import hashlib
import numpy as np


def main():
    # Instantiate retriever (MiniLM by default). This requires `sentence-transformers`.
    try:
        retriever = DenseRetriever('minilm')
        print('Loaded DenseRetriever (MiniLM)')
    except Exception as e:
        # Fallback: deterministic MockRetriever so the experiment can run without
        # heavy dependencies. This provides deterministic embeddings based on
        # SHA1 hashing of the text.
        print('DenseRetriever failed to load:', e)

        class MockRetriever:
            def __init__(self, dim: int = 384):
                self.dim = dim

            def _det_emb(self, text: str) -> np.ndarray:
                # deterministic pseudo-random vector from SHA1
                h = hashlib.sha1(text.encode()).digest()
                seed = int.from_bytes(h[:4], 'big')
                rng = np.random.RandomState(seed)
                v = rng.randn(self.dim).astype('float32')
                v /= np.linalg.norm(v) + 1e-8
                return v

            def encode_query(self, text: str) -> np.ndarray:
                return self._det_emb(text)

            def encode_document(self, text: str) -> np.ndarray:
                return self._det_emb(text)

            def encode_batch(self, texts, batch_size: int = 32):
                return np.vstack([self._det_emb(t) for t in texts]).astype('float32')

            def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
                return float(np.dot(a, b))

        retriever = MockRetriever(dim=384)
        print('Using MockRetriever fallback (deterministic embeddings)')
    try:
        kb = KnowledgeBase(retriever)
    except Exception as e:
        print('KnowledgeBase (FAISS) failed to load:', e)
        # Fallback: simple in-memory knowledge base using dot-product on
        # retriever embeddings. Deterministic and lightweight for testing.
        class MockKnowledgeBase:
            def __init__(self, retriever):
                self.retriever = retriever
                self.docs = []

            def add_documents(self, documents):
                for doc in documents:
                    d = dict(doc)
                    text = d.get('text', d.get('full_text', ''))
                    d['_emb'] = self.retriever.encode_document(text)
                    self.docs.append(d)

            def retrieve(self, query, k=5):
                q_emb = self.retriever.encode_query(query)
                sims = []
                for i, d in enumerate(self.docs):
                    emb = d.get('_emb')
                    score = float(np.dot(emb, q_emb))
                    sims.append((score, i))
                sims.sort(key=lambda x: x[0], reverse=True)
                results = []
                for score, idx in sims[:k]:
                    doc = dict(self.docs[idx])
                    doc['score'] = float(score)
                    doc['index'] = int(idx)
                    results.append(doc)
                return results

            def poison(self, adversarial_docs):
                self.add_documents(adversarial_docs)

            def reset(self):
                self.docs = []

            def get_document_by_id(self, doc_id):
                for d in self.docs:
                    if d.get('id') == doc_id:
                        return d
                return None

        kb = MockKnowledgeBase(retriever)
        print('Using MockKnowledgeBase fallback')
    dsrm = DSRMSimulator()

    # Wire NeighborGenerator to use the retriever for semantic filtering and
    # request the T5 paraphrase model (lazy-loaded). Model download requires
    # `transformers` and access to HuggingFace model weights.
    ng = NeighborGenerator(model_name=None, retriever=retriever)
    cq = CQRCDFilter(retriever, ng)

    # create some dummy legitimate docs
    legit_docs = []
    for i in range(5):
        legit_docs.append({'id': f'doc_leg_{i}', 'text': f'Legitimate document content about topic {i}', 'label': 'legitimate', 'tools_mentioned': ['RiskAssessmentTool']})
    kb.add_documents(legit_docs)

    # adversarial docs
    adv_docs = []
    for i in range(5):
        adv_docs.append(dsrm.generate_blackbox('Evaluate the risk and potential returns of investing in a new sector', f'AttackTool{i}', 'do bad', ['RiskAssessmentTool']))
    kb.add_documents(adv_docs)

    query = 'Evaluate the risk and potential returns of investing in a new sector'

    # Test neighbor generation (this will lazy-load T5 if available)
    try:
        paraphrases = ng.generate(query, n=5)
        print('NeighborGenerator model_loaded:', getattr(ng, '_model_loaded', False))
        print('Paraphrases:')
        for p in paraphrases:
            print(' -', p)
    except Exception as e:
        print('Neighbor generation failed:', e)

    retrieved = kb.retrieve(query, k=10)
    filtered, scores = cq.filter(query, retrieved, return_scores=True)
    print('Retrieved:', [d['id'] for d in retrieved])
    print('Scores:', scores)
    print('Filtered:', [d['id'] for d in filtered])


if __name__ == '__main__':
    main()
