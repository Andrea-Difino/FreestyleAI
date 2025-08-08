import pandas as pd
import regex as re
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
import pickle

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<START>", "<END>", "<UNK>", "<LINE>"]

CSV_PATH          = "FreestyleAI/updated_rappers.csv"     
CORPUS_PATH       = "tmp_corpus.txt"                      # file intermedio
SPM_MODEL_PREFIX  = "FreestyleAI/models/bpe_spm"          # model + vocab saranno salvati qui
VOCAB_SIZE        = 24000                                 # numero di token BPE (puoi cambiarlo)
CHAR_COVERAGE     = 1.0                                   # copertura caratteri Unicode (1.0 = tutti)


_SPLIT_RE = re.compile(GPT4_SPLIT_PATTERN)   # compilata una sola volta

# ----------------------------------------------------------------------
#   FUNZIONI DI SUPPORTO
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Rimuove spazi multipli e strip finale."""
    return re.sub(r'\s{2,}', ' ', text).strip()


def is_informative(word: str) -> bool:
    """Logica identica al tuo script originale."""
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
    """Yield di token usando la regex pre‑compilata."""
    return (m.group(0) for m in _SPLIT_RE.finditer(line))


def process_one_song(lyrics: list[str]) -> list[str]:
    """Applica la regex + aggiunge i token speciali."""
    tokens = ["<START>"]
    for line in lyrics:
        line = clean_text(line.lower())
        if not line:
            continue
        for w in split_line(line):
            tokens.append(w if is_informative(w) else "<UNK>")
        tokens.append("<LINE>")
    # L’ultimo <LINE> diventa <END>
    if tokens[-1] == "<LINE>":
        tokens[-1] = "<END>"
    else:
        tokens.append("<END>")
    return tokens


# ----------------------------------------------------------------------
#   Scrivi il corpus di testo per SentencePiece
# ----------------------------------------------------------------------
def write_corpus(csv_path: str, corpus_path: str) -> None:
    """
    Legge il CSV, tokenizza ogni canzone e scrive una riga
    (token separati da spazio) in `corpus_path`.
    """
    print("🔎  Lettura CSV …")
    db = pd.read_csv(
        csv_path,
        usecols=["song", "lyric"],
        engine="pyarrow",            # più veloce del default
    )

    # Raggruppa per canzone (come nel tuo script)
    grouped = db.groupby("song")["lyric"].apply(list)

    corpus_file = Path(corpus_path)
    with corpus_file.open("w", encoding="utf-8") as fout:
        for lyrics in tqdm(grouped, desc="Tokenizzazione canzoni"):
            song_tokens = process_one_song(lyrics)
            fout.write(" ".join(song_tokens) + "\n")
    print(f"✅ Corpus scritto in: {corpus_path}")


# ----------------------------------------------------------------------
#   Addestra SentencePiece (BPE)
# ----------------------------------------------------------------------
def train_spm(corpus_path: str,
              model_prefix: str,
              vocab_size: int,
              character_coverage: float = 1.0) -> str:
    """
    Esegue SentencePieceTrainer su `corpus_path`.
    Salva:
        - {model_prefix}.model
        - {model_prefix}.vocab
    Restituisce il percorso completo del file .model.
    """
    spm_cmd = (
        f"--input={corpus_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        "--model_type=bpe "
        "--pad_id=-1 "            # nessun token di padding (non serve per il tuo model)
        "--unk_id=0 "             # <UNK> sarà il token 0
        "--bos_id=1 "             # <START>
        "--eos_id=2 "             # <END>
        "--user_defined_symbols=<LINE> "   # aggiungiamo <LINE> come simbolo extra
        "--max_sentence_length=6000 "
    )
    print("🚀  Addestramento SentencePiece …")
    spm.SentencePieceTrainer.Train(spm_cmd)
    model_path = f"{model_prefix}.model"
    print(f"✅  Modello SentencePiece salvato in: {model_path}")
    return model_path


# ----------------------------------------------------------------------
#   Codifica l’intero dataset in una lista di ID
# ----------------------------------------------------------------------
def encode_dataset(csv_path: str, sp_model_path: str) -> list[int]:
    """
    Legge di nuovo il CSV, tokenizza con la regex e restituisce
    una lista piatta di interi (ID del vocabolo SentencePiece).
    Questo file può essere salvato una volta sola (`ids_spm.pkl`) e
    poi caricato direttamente dal training.
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    db = pd.read_csv(
        csv_path,
        usecols=["song", "lyric"],
        engine="pyarrow",
    )

    grouped = db.groupby("song")["lyric"].apply(list)

    all_ids = []
    for lyrics in tqdm(grouped, desc="Encoding dataset"):
        tokens = process_one_song(lyrics)          # <START> … <END> + <LINE>
        raw = " ".join(tokens)                     # SentencePiece legge token separati da spazio
        ids = sp.EncodeAsIds(raw)                  # restituisce BOS/EOS automaticamente
        all_ids.extend(ids)
    return all_ids


if __name__ == "__main__":
    csv_path   = CSV_PATH
    corpus_path = CORPUS_PATH
    model_prefix = SPM_MODEL_PREFIX
    vocab_sz   = VOCAB_SIZE

    # ---- Crea il corpus testuale ----
    write_corpus(csv_path, corpus_path)

    # ---- Addestra SentencePiece ----
    spm_model = train_spm(
        corpus_path=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_sz,
        character_coverage=CHAR_COVERAGE,
    )

    # ---- Codifica tutto il dataset in ID ----
    ids = encode_dataset(csv_path, spm_model)

    # Salviamo gli ID in formato pickle (puoi usare anche np.save se preferisci)
    ids_path = Path("FreestyleAI/data/ids_spm.pkl")
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    with ids_path.open("wb") as f:
        pickle.dump(ids, f)
    print(f"✅  Lista di ID salvata in: {ids_path}")

    Path(corpus_path).unlink(missing_ok=True)
    print("🗑️  Corpus temporaneo cancellato.")