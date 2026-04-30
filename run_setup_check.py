"""Validate the local CQRCD RTX setup."""
from __future__ import annotations


def main() -> None:
    import bitsandbytes
    import faiss
    import torch
    import transformers
    import sentence_transformers

    print('=== CQRCD Setup Check ===')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        print(f'VRAM reserved: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB')
    print(f'Transformers: {transformers.__version__}')
    print(f'SentenceTransformers: {sentence_transformers.__version__}')
    print(f'FAISS GPU available: {hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0}')
    print(f'bitsandbytes: {bitsandbytes.__version__}')
    print('=== All checks passed ===')


if __name__ == '__main__':
    main()
