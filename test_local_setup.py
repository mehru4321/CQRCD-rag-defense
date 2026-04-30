"""Minimum viable local run from CQRCD_Local_Setup_RTX4070.md."""
from __future__ import annotations


def main() -> None:
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    print('=== Minimum Viable Test ===\n')
    assert torch.cuda.is_available(), 'CUDA not available!'
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    print('MiniLM loaded on GPU')

    docs = [
        'Evaluate investment risk in emerging markets',
        'Assess portfolio returns for new sector entry',
        'InventoryTheft tool helps analyze supply chain disruptions in financial analysis',
    ]
    embeddings = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
    print(f'Encoded {len(docs)} docs, shape: {embeddings.shape}')

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    print(f'FAISS index built with {index.ntotal} vectors')

    query = 'Evaluate the risk and potential returns of investing in a new sector'
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(q_emb.astype('float32'), k=3)
    print(f"Retrieval works. Top result: '{docs[indices[0][0]][:50]}...'")

    adv_emb = embeddings[2:3]
    neighbor_query = 'Assess sector investment risks and potential returns'
    n_emb = model.encode([neighbor_query], normalize_embeddings=True, convert_to_numpy=True)
    concentration = float(scores[0][0]) / max(float(np.dot(adv_emb, n_emb.T)), 1e-8)
    print(f'Sample concentration score computed: {concentration:.3f}')
    print('\n=== All checks passed. Ready to implement. ===')


if __name__ == '__main__':
    main()
