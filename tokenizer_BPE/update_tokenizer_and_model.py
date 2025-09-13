import argparse
import pandas as pd
from pathlib import Path
import regex as re
import sentencepiece as spm
import torch
from FreestyleAI import WordGramModel #type: ignore

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
_SPLIT_RE = re.compile(GPT4_SPLIT_PATTERN)

# --------------------- Helper SentencePiece ---------------------
def _sp_piece_size(sp):
    if hasattr(sp, "GetPieceSize"):
        return sp.GetPieceSize()
    if hasattr(sp, "get_piece_size"):
        return sp.get_piece_size()
    raise AttributeError("SentencePieceProcessor GetPieceSize method not found.")

def _sp_id_to_piece(sp, idx: int) -> str:
    if hasattr(sp, "id_to_piece"):
        return sp.id_to_piece(idx)
    if hasattr(sp, "IdToPiece"):
        return sp.IdToPiece(idx)
    raise AttributeError("SentencePieceProcessor id_to_piece method not found.")

# --------------------- Remap embeddings ---------------------
def remap_and_expand_model_by_pieces(model, sp_old, sp_new, device):
    old_vocab = _sp_piece_size(sp_old)
    new_vocab = _sp_piece_size(sp_new)

    if new_vocab == old_vocab:
        print("ℹ️ Tokenizer vocab unchanged; no remap needed")
        return model, new_vocab

    print(f"🧭 Remapping embeddings: {old_vocab} → {new_vocab}")

    old_piece_to_id = { _sp_id_to_piece(sp_old, i): i for i in range(old_vocab) }

    new_model = WordGramModel(new_vocab).to(device)

    with torch.no_grad():
        # Init embeddings/output
        torch.nn.init.normal_(new_model.token_embedding_table.weight, 0.0, 0.02)
        torch.nn.init.normal_(new_model.output[1].weight, 0.0, 0.02)
        torch.nn.init.zeros_(new_model.output[1].bias)

        # Copy non-vocab layers
        for name, param in model.named_parameters():
            if "token_embedding_table" in name or "output.1" in name:
                continue
            new_model.state_dict()[name].copy_(param)

        # Copy embeddings/output per token esistente
        copied = 0
        for new_id in range(new_vocab):
            piece = _sp_id_to_piece(sp_new, new_id)
            old_id = old_piece_to_id.get(piece, None)
            if old_id is not None:
                new_model.token_embedding_table.weight[new_id] = model.token_embedding_table.weight[old_id]
                new_model.output[1].weight[new_id] = model.output[1].weight[old_id]
                new_model.output[1].bias[new_id] = model.output[1].bias[old_id]
                copied += 1

    print(f"✅ Copiati {copied} token; nuovi token: {new_vocab - copied}")
    return new_model, new_vocab

def clean_text(text: str) -> str:
    # normalizza spazi multipli e trim
    return re.sub(r"\s+", " ", str(text)).strip()

def is_informative(word: str) -> bool:
    w = word.strip()
    if not w:
        return False
    if w.lower() in {"<unk>", "<line>", "<start>", "<end>"}:
        return False
    if w.isdigit():
        return True
    if re.fullmatch(r"[a-zA-Z']+", w):
        return True
    return w == " "


def split_line(line: str):
    return (m.group(0) for m in _SPLIT_RE.finditer(line))

def build_corpus(old_csv, new_csv, combined_corpus_path):
    print("📚 Creazione corpus combinato...")

    # === Vecchio dataset ===
    df_old = pd.read_csv(old_csv, usecols=["song", "lyric"])
    grouped_old = df_old.groupby("song")["lyric"].apply(list)

    old_lines = []
    for lyrics in grouped_old:
        tokens = ["<START>"]
        for i, line in enumerate(lyrics):
            line = clean_text(line.lower())
            if not line:
                continue

            # tokenizza e filtra token non informativi
            for w in split_line(line):
                tokens.append(w if is_informative(w) else "<UNK>")

            if i < len(lyrics) - 1:
                tokens.append("<LINE>")
        tokens.append("<END>")
        old_lines.append(" ".join(tokens) + "\n")

    # === Nuovo dataset ===
    df_new = pd.read_csv(new_csv, usecols=["battle_id", "bar"])
    grouped_new = df_new.groupby("battle_id")["bar"].apply(list)

    new_lines = []
    for bars in grouped_new:
        tokens = ["<START>"]
        for i, bar in enumerate(bars):
            bar = clean_text(bar.lower())
            if not bar:
                continue

            for w in split_line(bar):
                tokens.append(w if is_informative(w) else "<UNK>")

            if i < len(bars) - 1:
                tokens.append("<LINE>")
        tokens.append("<END>")
        new_lines.append(" ".join(tokens) + "\n")

    # === Scrittura corpus combinato ===
    with open(combined_corpus_path, "w", encoding="utf-8") as fout:
        fout.writelines(old_lines + new_lines)

    print(f"✅ Corpus salvato in {combined_corpus_path}")


# --------------------- Train SentencePiece ---------------------
def train_new_spm(corpus_path, model_prefix, vocab_size):
    cmd = (
        f"--input={corpus_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        "--model_type=bpe "
        "--unk_id=0 "
        "--bos_id=-1 "
        "--eos_id=-1 "             
        "--pad_id=1 "        
        "--user_defined_symbols=<START>,<END>,<UNK>,<LINE> "   
        "--max_sentence_length=5000 "
    )
    print("🚀 Riallenamento SentencePiece...")
    spm.SentencePieceTrainer.Train(cmd)
    return f"{model_prefix}.model"

# --------------------- MAIN ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-spm", default="FreestyleAI/models/bpe_spm.model")
    parser.add_argument("--old-model", default="FreestyleAI/models/bpe-model.pt")
    parser.add_argument("--old-csv", default="FreestyleAI/updated_rappers.csv")
    parser.add_argument("--new-csv", default="FreestyleAI/dataset_creation/freestyle_dataset_kotd_clean.csv")
    parser.add_argument("--out-prefix", default="FreestyleAI/models/bpe_spm_updated")
    parser.add_argument("--out-model", default="FreestyleAI/models/bpe-model-updated.pt")
    parser.add_argument("--out-metadata", default="FreestyleAI/metadata/bpe-metadata-updated.pt")
    parser.add_argument("--vocab-size", type=int, default=34000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    combined_corpus = "FreestyleAI/dataset_creation/combined_corpus.txt"

    build_corpus(args.old_csv, args.new_csv, combined_corpus)
    new_spm_path = train_new_spm(combined_corpus, args.out_prefix, args.vocab_size)

    sp_old = spm.SentencePieceProcessor(); sp_old.Load(args.old_spm)
    sp_new = spm.SentencePieceProcessor(); sp_new.Load(new_spm_path)

    # Carica vecchio modello
    old_model = WordGramModel(_sp_piece_size(sp_old)).to(DEVICE)
    old_model.load_state_dict(torch.load(args.old_model, map_location=DEVICE))

    # Remap embeddings e output
    new_model, final_vocab_size = remap_and_expand_model_by_pieces(old_model, sp_old, sp_new, DEVICE)
    torch.save(new_model.state_dict(), args.out_model)

    # Salva metadata
    new_metadata = {
        "sp_model_path": new_spm_path,
        "vocab_size": final_vocab_size,
        "original_vocab_size": _sp_piece_size(sp_old),
        "base_model": Path(args.old_model).name,
        "updated_with": Path(args.new_csv).name,
    }
    torch.save(new_metadata, args.out_metadata)

    print("\n✅ Aggiornamento completato!")
    print(f"📁 Nuovo tokenizer: {new_spm_path}")
    print(f"📁 Nuovo modello PyTorch: {args.out_model}")
    print(f"📁 Nuovo metadata: {args.out_metadata}")
    print(f"🔠 Vocabolario finale: {final_vocab_size}")

if __name__ == "__main__":
    main()
