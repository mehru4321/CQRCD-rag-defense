"""Project-wide configuration and hyperparameters for CQRCD.

Defaults follow ``CQRCD_Local_Setup_RTX4070.md`` while keeping CPU fallbacks
for development machines without CUDA.
"""
from __future__ import annotations

import random
import numpy as _np

# Device configuration
try:
    import torch as _torch
except Exception:  # pragma: no cover - torch may be unavailable during docs checks
    _torch = None

DEVICE = 'cuda' if _torch is not None and _torch.cuda.is_available() else 'cpu'
VRAM_GB = (
    _torch.cuda.get_device_properties(0).total_memory / 1e9
    if DEVICE == 'cuda'
    else 0
)

# RTX 4070 model loading strategy
LLM_LOAD_IN_4BIT = True
LLM_MAX_MEMORY = {0: '6GB', 'cpu': '24GB'}
FAISS_USE_GPU = DEVICE == 'cuda'
FAISS_GPU_MEMORY_MB = 512
RETRIEVER_BATCH_SIZE = 64
DPR_BATCH_SIZE = 32
PARAPHRASE_BATCH_SIZE = 8
LLM_BATCH_SIZE = 1

# Retrieval
DEFAULT_RETRIEVER = 'minilm'    # 'dpr', 'minilm', 'realm'
TOP_K = 5                        # Number of documents retrieved per query

# CQRCD
# Threshold recalibrated for MiniLM embedding space (Session 004).
# The paper's τ_C = 1.65 was tuned for DPR where paraphrase-to-query cosine
# similarity is ~0.50-0.60.  MiniLM compresses the similarity range to
# ~0.82-0.92, so the maximum achievable concentration for black-box adversarial
# documents (text = exact query) is ~1.0 / 0.85 ≈ 1.18-1.25.  A threshold of
# 1.20 sits in the validated separating region between legitimate (mean ~1.05)
# and adversarial (mean ~1.22-1.38) distributions.
CONCENTRATION_THRESHOLD = 1.20   # τ_C — documents above this are flagged
N_NEIGHBORS = 5                  # Number of query paraphrases generated
EPSILON = 1e-8                   # Division-by-zero guard

# DSRM Attack (Simulator)
SRM_SIMILARITY_THRESHOLD = 0.6
CSRM_REASONING_LENGTH = 45
WHITEBOX_OPTIM_STEPS = 30
N_NEGATIVES = 40
RANDOM_SEED = 42

# Agent
MAX_TOKENS = 512

# Evaluation
N_ASB_TASKS = 10
N_ATTACK_SCENARIOS = 400
VALIDATION_SPLIT = 0.2

# Paraphrase
PARAPHRASE_NUM_BEAMS = 10
PARAPHRASE_MAX_LENGTH = 128

random.seed(RANDOM_SEED)
_np.random.seed(RANDOM_SEED)
if _torch is not None:
    _torch.manual_seed(RANDOM_SEED)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(RANDOM_SEED)

__all__ = [
    'DEVICE','VRAM_GB','LLM_LOAD_IN_4BIT','LLM_MAX_MEMORY','FAISS_USE_GPU',
    'FAISS_GPU_MEMORY_MB','RETRIEVER_BATCH_SIZE','DPR_BATCH_SIZE',
    'PARAPHRASE_BATCH_SIZE','LLM_BATCH_SIZE',
    'DEFAULT_RETRIEVER','TOP_K','CONCENTRATION_THRESHOLD','N_NEIGHBORS','EPSILON',
    'SRM_SIMILARITY_THRESHOLD','CSRM_REASONING_LENGTH','WHITEBOX_OPTIM_STEPS','N_NEGATIVES',
    'RANDOM_SEED','MAX_TOKENS','N_ASB_TASKS','N_ATTACK_SCENARIOS','VALIDATION_SPLIT',
    'PARAPHRASE_NUM_BEAMS','PARAPHRASE_MAX_LENGTH'
]
