"""GPU memory helpers for local RTX development."""
from __future__ import annotations

import gc


def clear_gpu_memory() -> None:
    """Release unused CUDA memory between major experiments/model swaps."""
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_usage() -> str:
    """Return a compact VRAM usage string for logs."""
    try:
        import torch
    except Exception:
        return 'VRAM: torch unavailable'
    if not torch.cuda.is_available():
        return 'VRAM: CUDA unavailable'
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f'VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total'


class ModelManager:
    """Context manager that unloads one model family before another is loaded."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None

    def __enter__(self):
        self.model = self._load(self.model_type)
        return self.model

    def __exit__(self, *args):
        del self.model
        self.model = None
        clear_gpu_memory()

    def _load(self, model_type: str):
        from modules.llm_loader import load_llm_4bit, load_paraphrase_model, load_retriever

        if model_type == 'llm':
            return load_llm_4bit()
        if model_type == 'retriever':
            return load_retriever()
        if model_type == 'paraphrase':
            return load_paraphrase_model()
        raise ValueError(f'Unknown model_type: {model_type}')
