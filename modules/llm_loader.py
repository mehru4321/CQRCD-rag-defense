"""Model loading helpers tuned for the RTX 4070 local setup."""
from __future__ import annotations

from config import DEVICE, LLM_LOAD_IN_4BIT, LLM_MAX_MEMORY


def load_llm_4bit(model_name: str = 'Qwen/Qwen2-7B-Instruct') -> tuple:
    """Load an instruction LLM with 4-bit quantization when CUDA is available."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if DEVICE == 'cuda' and LLM_LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            max_memory=LLM_MAX_MEMORY,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(DEVICE)

    model.eval()
    return model, tokenizer


def load_retriever(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """Load the lightweight retriever onto the configured device."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=DEVICE)


def load_paraphrase_model(model_name: str = 'Vamsi/T5_Paraphrase_Paws') -> tuple:
    """Load the T5 paraphrase model onto the configured device."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {}
    if DEVICE == 'cuda':
        kwargs['torch_dtype'] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer
