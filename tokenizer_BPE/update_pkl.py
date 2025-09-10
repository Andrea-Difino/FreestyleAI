import pickle
import sentencepiece as spm
from pathlib import Path

# Percorsi
SPM_MODEL_PATH = "FreestyleAI/models/bpe_spm_updated.model"   # nuovo tokenizer creato
CORPUS_PATH = Path("FreestyleAI/dataset_creation/combined_corpus.txt")         # corpus combinato salvato dallo script di aggiornamento
OUT_PKL_PATH = Path("FreestyleAI/data/ids_spm_updated.pkl")

def main():
    # Carica tokenizer aggiornato
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL_PATH)

    # Carica corpus in memoria
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        text = f.read()

    # Tokenizza in ID interi
    ids = sp.EncodeAsIds(text)

    print(f"📦 Numero token generati: {len(ids):,}")

    # Salva come pickle
    OUT_PKL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PKL_PATH.open("wb") as f:
        pickle.dump(ids, f)

    print(f"✅ Salvato nuovo file: {OUT_PKL_PATH}")

if __name__ == "__main__":
    main()
