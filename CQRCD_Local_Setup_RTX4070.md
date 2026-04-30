# CQRCD: Local Development Setup
## RTX 4070 (8GB VRAM) + 32GB System RAM

---

## SUPPLEMENT TO: CQRCD_Implementation_Master.md
## Replace all Colab-specific instructions with this document.
## Everything else in the master document remains valid.

---

## 1. WHY LOCAL IS BETTER FOR THIS PROJECT

Running locally on your RTX 4070 is the recommended approach over Colab for the
following concrete reasons:

1. **No session timeouts** — Colab disconnects after 90 minutes of idle. Your ablation
   study (Experiment 5) runs 5 variables × multiple values = potentially 2-3 hours. This
   will disconnect on Colab. Locally, it runs uninterrupted.

2. **32GB system RAM** — Colab gives ~12GB system RAM. FAISS indexes, embedding caches,
   and result DataFrames for 400 attack scenarios will comfortably fit in your 32GB.
   On Colab you would need to carefully manage memory and flush caches between experiments.

3. **Persistent storage** — On Colab, if you forget to mount Drive or the session crashes,
   you lose generated adversarial documents, cached embeddings, and intermediate results.
   Locally, everything persists.

4. **Full debugging** — You can use VS Code with the Python debugger, set breakpoints inside
   cqrcd_filter.py, inspect tensors, and step through the concentration score computation.
   This is essential when calibrating the threshold τ_C.

5. **VRAM trade-off is manageable** — You have 8GB VRAM vs Colab T4's 15GB. The only
   model that strains your VRAM is the LLM backbone. With 4-bit quantization via
   bitsandbytes, LLaMA-3-8B fits in ~4.5GB VRAM, leaving 3.5GB for retriever models
   and batch operations. This is sufficient.

---

## 2. ENVIRONMENT SETUP

### 2.1 Prerequisites

Check these first:

```bash
# Check CUDA version (must be 11.8 or 12.x)
nvidia-smi

# Check Python version (must be 3.10 or 3.11)
python --version
```

If CUDA is not installed: https://developer.nvidia.com/cuda-downloads
If Python is wrong version: use pyenv or conda to manage versions

### 2.2 Create Virtual Environment

```bash
# Option A: conda (recommended — handles CUDA-linked packages better)
conda create -n cqrcd python=3.10
conda activate cqrcd

# Option B: venv
python -m venv cqrcd_env
# Windows:
cqrcd_env\Scripts\activate
# Linux/Mac:
source cqrcd_env/bin/activate
```

### 2.3 Install PyTorch with CUDA Support

This step is critical. Do NOT use pip install torch directly — it may install CPU-only.

```bash
# For CUDA 12.1 (check your nvidia-smi output for CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected output:
# True
# NVIDIA GeForce RTX 4070 Laptop GPU
```

### 2.4 Install All Dependencies

```bash
# Core ML libraries
pip install transformers>=4.40.0
pip install sentence-transformers>=2.6.0
pip install accelerate>=0.25.0
pip install bitsandbytes>=0.43.0   # CRITICAL for 4-bit quantization

# Vector store
pip install faiss-gpu              # GPU-accelerated FAISS (since you have CUDA)
# NOTE: if faiss-gpu fails, fall back to:
# pip install faiss-cpu

# Data and evaluation
pip install numpy scipy scikit-learn pandas tqdm
pip install datasets rouge-score nltk

# Visualization
pip install matplotlib seaborn

# Hugging Face hub (for model downloads)
pip install huggingface-hub

# Verify bitsandbytes works with your GPU
python -c "import bitsandbytes as bnb; print('bitsandbytes OK')"
```

### 2.5 Verify Full Setup

```python
# run_setup_check.py — run this before starting implementation
import torch
import transformers
import sentence_transformers
import faiss
import bitsandbytes

print("=== CQRCD Setup Check ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"VRAM free: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserved")
print(f"Transformers: {transformers.__version__}")
print(f"FAISS GPU: {'gpu' in dir(faiss)}")
print("=== All checks passed ===")
```

---

## 3. VRAM MANAGEMENT STRATEGY

8GB VRAM requires careful model loading strategy. The key rule:
**Never load the LLM and the retriever simultaneously in VRAM.**

### 3.1 VRAM Budget

| Component | VRAM Usage | When Active |
|-----------|-----------|-------------|
| LLaMA-3-8B (4-bit) | ~4.5 GB | Agent simulation (Exp 3 only) |
| Qwen2-7B (4-bit) | ~4.2 GB | Agent simulation (Exp 3 only) |
| DPR (query + context encoder) | ~0.9 GB | All experiments |
| MiniLM | ~0.1 GB | All experiments |
| ReaLM | ~0.05 GB | All experiments |
| T5-Paraphrase | ~0.3 GB | All experiments |
| FAISS GPU index | ~0.1 GB | All experiments |
| PyTorch overhead | ~0.5 GB | Always |

**Peak usage scenarios:**

```
Experiments 1, 2, 4, 5 (no LLM needed):
DPR + T5 + FAISS + overhead = ~1.9 GB ✅ Fits easily

Experiment 3 (with LLM backbone):
LLM (4-bit) + MiniLM + T5 + overhead = ~5.2 GB ✅ Fits in 8GB
BUT: Do NOT load DPR simultaneously with LLM
```

### 3.2 Model Loading Pattern

Replace the Colab-style "load everything once" pattern with explicit device management:

```python
# config.py — ADD THESE SETTINGS

import torch

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

# Model loading strategy
# For 8GB VRAM: never load LLM + DPR simultaneously
LLM_LOAD_IN_4BIT = True          # Always True for 8GB VRAM
LLM_MAX_MEMORY = {0: "6GB", "cpu": "24GB"}  # Leave 2GB buffer, spill to RAM if needed

# FAISS: use GPU index for main experiments, CPU for ablation (saves VRAM)
FAISS_USE_GPU = True
FAISS_GPU_MEMORY_MB = 512        # Limit FAISS GPU memory to 512MB

# Batch sizes calibrated for 8GB VRAM
RETRIEVER_BATCH_SIZE = 64        # MiniLM: can handle large batches
DPR_BATCH_SIZE = 32              # DPR is larger
PARAPHRASE_BATCH_SIZE = 8        # T5 is moderate
LLM_BATCH_SIZE = 1               # LLM in 4-bit: process one at a time
```

### 3.3 GPU Memory Cleanup Pattern

Use this pattern between experiments to prevent VRAM accumulation:

```python
# utils/gpu_utils.py

import torch
import gc

def clear_gpu_memory():
    """Call this between major experiments or when switching models."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_vram_usage() -> str:
    """Returns formatted VRAM usage string for logging."""
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total"

class ModelManager:
    """
    Context manager for loading models.
    Automatically unloads models when done to free VRAM.

    Usage:
        with ModelManager('llm') as llm:
            results = agent.run_with_llm(llm, dataset)
        # LLM is unloaded here, VRAM freed
        with ModelManager('retriever') as ret:
            embeddings = ret.encode_batch(documents)
    """

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None

    def __enter__(self):
        self.model = self._load(self.model_type)
        return self.model

    def __exit__(self, *args):
        del self.model
        clear_gpu_memory()

    def _load(self, model_type: str):
        if model_type == 'llm':
            return load_llm_4bit()
        elif model_type == 'retriever':
            return load_retriever()
        elif model_type == 'paraphrase':
            return load_paraphrase_model()
```

---

## 4. UPDATED MODEL LOADING CODE

### 4.1 LLM Loading (4-bit quantization for 8GB VRAM)

```python
# modules/llm_loader.py

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_llm_4bit(model_name: str = "Qwen/Qwen2-7B-Instruct") -> tuple:
    """
    Loads LLM in 4-bit quantization.
    Uses Qwen2-7B as default (no gating, freely downloadable).
    For LLaMA-3-8B: requires HuggingFace account + accepting Meta license.

    VRAM usage:
    - Qwen2-7B (4-bit):   ~4.2 GB
    - LLaMA-3-8B (4-bit): ~4.5 GB

    Returns: (model, tokenizer)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,    # saves ~0.4GB extra
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                 # auto distributes across GPU + CPU if needed
        max_memory={0: "6GB", "cpu": "24GB"},
        torch_dtype=torch.float16
    )
    model.eval()

    print(f"LLM loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    return model, tokenizer


def load_retriever(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads retriever model onto GPU.
    MiniLM uses ~0.1GB VRAM — negligible.
    DPR uses ~0.9GB VRAM — load only when LLM is NOT loaded.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device='cuda')
    return model


def load_paraphrase_model(model_name: str = "Vamsi/T5_Paraphrase_Paws"):
    """
    Loads T5 paraphrase model.
    Uses ~0.3GB VRAM.
    Safe to keep loaded alongside retriever models.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    model.eval()
    return model, tokenizer
```

### 4.2 FAISS with GPU

```python
# In knowledge_base.py — replace faiss-cpu code with this

import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """
    Builds FAISS flat index.
    On RTX 4070: use GPU index for ~5x faster search vs CPU.
    """
    dim = embeddings.shape[1]

    # Build on CPU first
    index = faiss.IndexFlatIP(dim)   # Inner product (= cosine on normalized vectors)
    index.add(embeddings)

    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024)  # 512MB VRAM for FAISS
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        return gpu_index

    return index  # fallback to CPU if GPU unavailable
```

---

## 5. UPDATED BATCH SIZES AND PERFORMANCE ESTIMATES

On RTX 4070 locally, you can use larger batch sizes than Colab because:
1. No shared GPU (Colab T4 is shared in free tier)
2. Faster PCIe connection to system RAM (data loading is faster)

```python
# Revised batch sizes for RTX 4070

# Encoding documents with MiniLM
# Colab T4: batch_size=16 (safe)
# RTX 4070: batch_size=64 (fine, ~0.1GB VRAM peak)
RETRIEVER_BATCH_SIZE = 64

# Encoding with DPR
# Colab T4: batch_size=8
# RTX 4070: batch_size=32
DPR_BATCH_SIZE = 32

# Paraphrase generation (T5)
# Colab T4: batch_size=4
# RTX 4070: batch_size=8
PARAPHRASE_BATCH_SIZE = 8

# LLM inference (4-bit, 8GB VRAM)
# Both platforms: batch_size=1 (4-bit models don't benefit from batching much)
LLM_BATCH_SIZE = 1
```

**Estimated runtimes on RTX 4070 (vs Colab T4):**

| Task | Colab T4 | RTX 4070 |
|------|----------|----------|
| Encode 400 docs (MiniLM) | ~8 sec | ~4 sec |
| Generate 50 paraphrases (T5) | ~40 sec | ~20 sec |
| FAISS search (400 queries, K=5) | ~2 sec | ~1 sec |
| Concentration scores (400 docs, n=5) | ~90 sec | ~45 sec |
| Full Experiment 1 | ~10 min | ~5 min |
| Full Experiment 3 (mock LLM) | ~25 min | ~12 min |
| Full ablation study | ~3 hours | ~1.5 hours |

---

## 6. RECOMMENDED LOCAL DEVELOPMENT WORKFLOW

### 6.1 IDE Setup

Use **VS Code** with these extensions:
- Python (Microsoft)
- Pylance
- Jupyter (for .ipynb files)
- GitLens

In VS Code settings, configure the Python interpreter to your conda/venv:
```
Ctrl+Shift+P → "Python: Select Interpreter" → select cqrcd env
```

### 6.2 Project Initialization

```bash
# Clone or create project directory
mkdir cqrcd && cd cqrcd

# Initialize git (important — commit after each working module)
git init
git add .
git commit -m "initial: project structure"

# Create .gitignore
echo "*.pt\n*.bin\n__pycache__\n.env\nresults/\ndata/adversarial*\n.cache/" > .gitignore
```

### 6.3 HuggingFace Model Cache

Models will download to `~/.cache/huggingface/hub/` by default.
This can fill your C: drive quickly. Redirect to a drive with more space:

```bash
# Windows: set in environment variables or .env file
set HF_HOME=D:\huggingface_cache

# Linux/Mac:
export HF_HOME=/path/to/large/drive/.cache/huggingface
```

**Models to pre-download before starting (total ~15GB):**
```python
# download_models.py — run once
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

print("Downloading MiniLM...")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Downloading T5 paraphrase...")
AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

print("Downloading DPR...")
from transformers import DPRQuestionEncoder, DPRContextEncoder
DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

print("Downloading Qwen2-7B (this is large ~14GB)...")
# Only if using real LLM; skip if using mock LLM
# AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
# (model weights downloaded separately with load_llm_4bit())

print("All models cached.")
```

### 6.4 Running Experiments

```bash
# Always activate environment first
conda activate cqrcd

# Run experiments in order
python experiments/run_sanity_check.py          # Exp 1: ~5 min
python experiments/run_roc_analysis.py          # Exp 2: ~10 min
python experiments/run_main_experiment.py       # Exp 3: ~12 min
python experiments/run_adaptive_attack.py       # Exp 4: ~20 min
python experiments/run_ablation.py              # Exp 5: ~1.5 hours

# Generate all figures
python visualization/plot_results.py

# Check results
ls results/figures/   # Should have 6 PNG files at 400 DPI
ls results/tables/    # Should have CSV tables
```

---

## 7. THINGS TO CHANGE IN THE MASTER DOCUMENT

The following specific lines in `CQRCD_Implementation_Master.md` should be updated
when working locally:

| Master Doc Reference | Colab Version | Local RTX 4070 Version |
|---------------------|---------------|----------------------|
| Section 3.4 Dependency List | `faiss-cpu` | `faiss-gpu` |
| Section 5.2 KnowledgeBase | `faiss.IndexFlatIP` on CPU | `faiss.index_cpu_to_gpu()` |
| Section 5.3 DSRM white-box note | "embedding perturbation approximation is sufficient" | Can attempt full HotFlip with gradient computation — 30 steps takes ~2 min/document locally |
| Section 5.6 Agent Simulator | "Use MockLLM to avoid LLM inference cost" | Still recommended for bulk evaluation (400 scenarios); use real LLM only for case study demonstration |
| Section 6 Experiments | Runtimes assume Colab | See Section 5 of this document for revised runtimes |
| Section 8 config.py RETRIEVER_BATCH_SIZE | 16 | 64 |
| All `device='cpu'` references | CPU fallback | `device='cuda'` |

---

## 8. WHAT DOES NOT CHANGE

Everything in the master document that is not Colab-specific stays exactly the same:
- All module class signatures and method specifications
- All mathematical formulas (concentration score, contrastive loss)
- All hyperparameter values from the base paper (τ=0.6, L=45, K=5, seed=42)
- All data formats (JSON structures)
- All experiment designs and success criteria
- The 400 DPI figure requirement and matplotlib settings
- The implementation priority order (weeks 1-5)
- The paper claim mapping (Table in Section 14)

---

## 9. MINIMUM VIABLE FIRST RUN

Before building the full system, confirm your GPU setup works end-to-end
with this 5-minute test:

```python
# test_local_setup.py

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

print("=== Minimum Viable Test ===\n")

# 1. Confirm GPU
assert torch.cuda.is_available(), "CUDA not available!"
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# 2. Load retriever on GPU
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
print(f"✓ MiniLM loaded on GPU")

# 3. Encode sample documents
docs = [
    "Evaluate investment risk in emerging markets",
    "Assess portfolio returns for new sector entry",
    "InventoryTheft tool helps analyze supply chain disruptions in financial analysis"  # fake adversarial
]
embeddings = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
print(f"✓ Encoded {len(docs)} docs, shape: {embeddings.shape}")

# 4. Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"✓ FAISS index built with {index.ntotal} vectors")

# 5. Retrieve
query = "Evaluate the risk and potential returns of investing in a new sector"
q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
scores, indices = index.search(q_emb, k=3)
print(f"✓ Retrieval works. Top result: '{docs[indices[0][0]][:50]}...'")

# 6. Compute a concentration score manually
sim_target = float(scores[0][0])           # sim(adv_doc, query)
# For adversarial doc (index 2):
adv_emb = embeddings[2:3]
neighbor_query = "Assess sector investment risks and potential returns"
n_emb = model.encode([neighbor_query], normalize_embeddings=True, convert_to_numpy=True)
sim_neighbor = float(np.dot(adv_emb, n_emb.T))
concentration = sim_target / max(sim_neighbor, 1e-8)
print(f"✓ Sample concentration score computed: {concentration:.3f}")
print(f"  (Expected > 1.5 for adversarial-style doc)")

print("\n=== All checks passed. Ready to implement. ===")
```

Run this. If it completes without errors and prints all 7 checkmarks, your environment
is correctly configured for the full CQRCD implementation.

---

*End of Local Setup Supplement*
*RTX 4070 (8GB VRAM) + 32GB System RAM*
*Supplement to: CQRCD_Implementation_Master.md*
