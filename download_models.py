"""Pre-download the models used by the local RTX CQRCD workflow."""
from __future__ import annotations


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

    print('Downloading MiniLM...')
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print('Downloading T5 paraphrase...')
    AutoTokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')
    AutoModelForSeq2SeqLM.from_pretrained('Vamsi/T5_Paraphrase_Paws')

    print('Downloading DPR...')
    DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    print('Qwen2-7B is downloaded on first 4-bit load via modules.llm_loader.load_llm_4bit().')
    print('All lightweight models cached.')


if __name__ == '__main__':
    main()
