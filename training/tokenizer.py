import pandas as pd
import regex as re
import sentencepiece as spm
from pathlib import Path
from tqdm import tqdm
import pickle

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<START>", "<END>", "<UNK>", "<LINE>"]

CSV_PATH          = "FreestyleAI/updated_rappers.csv"     
CORPUS_PATH       = "FreestyleAI/tmp_corpus.txt"                      
SPM_MODEL_PREFIX  = "FreestyleAI/models/bpe_spm"          
VOCAB_SIZE        = 30000                                 # BPE token number
CHAR_COVERAGE     = 1.0                                   


_SPLIT_RE = re.compile(GPT4_SPLIT_PATTERN)   # compile one time

# ----------------------------------------------------------------------
#   SUPPORT FUNCTIONS
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    return re.sub(r'\s{2,}', ' ', text).strip()


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


def process_one_song(lyrics: list[str]) -> list[str]:
    """Apply regex + add special token."""
    tokens = ["<START>"]
    for i, line in enumerate(lyrics):
        line = clean_text(line.lower())
        if not line:
            continue
        for w in split_line(line):
            tokens.append(w if is_informative(w) else "<UNK>")

        tokens.append("<LINE>")
    # last <LINE> become <END>
    if tokens[-1] == "<LINE>":
        tokens[-1] = "<END>"
    else:
        tokens.append("<END>")
    return tokens


# ----------------------------------------------------------------------
#   Write corpus text for sentencepiece
# ----------------------------------------------------------------------
def write_corpus(csv_path: str, corpus_path: str) -> None:
    """
    Read the CSV, tokenize every songs a write a line
    (token separati da spazio) in `corpus_path`.
    """
    print("🔎  Lettura CSV …")
    db = pd.read_csv(
        csv_path,
        usecols=["song", "lyric"],
        engine="pyarrow",            
    )

    # groub by song
    grouped = db.groupby("song")["lyric"].apply(list)

    corpus_file = Path(corpus_path)
    with corpus_file.open("w", encoding="utf-8") as fout:
        for lyrics in tqdm(grouped, desc="Tokenizzazione canzoni"):
            song_tokens = process_one_song(lyrics)
            fout.write(" ".join(song_tokens) + "\n")
    print(f"✅ Corpus scritto in: {corpus_path}")


# ----------------------------------------------------------------------
#   train SentencePiece (BPE)
# ----------------------------------------------------------------------
def train_spm(corpus_path: str,
              model_prefix: str,
              vocab_size: int,
              character_coverage: float = 1.0) -> str:
    """
    Run SentencePieceTrainer on `corpus_path`.
    Save:
        - {model_prefix}.model
        - {model_prefix}.vocab
    Return the path of .model.
    """
    spm_cmd = (
        f"--input={corpus_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage={character_coverage} "
        "--model_type=bpe "         
        "--unk_id=0 "
        "--bos_id=-1 "
        "--eos_id=-1 "             
        "--pad_id=1 "        
        "--user_defined_symbols=<START>,<END>,<UNK>,<LINE> "   
        "--max_sentence_length=5000 "
    )
    print("🚀  Addestramento SentencePiece …")
    spm.SentencePieceTrainer.Train(spm_cmd)
    model_path = f"{model_prefix}.model"
    print(f"✅  Modello SentencePiece salvato in: {model_path}")
    return model_path


# ----------------------------------------------------------------------
#   Codify entire dataset into an ID list
# ----------------------------------------------------------------------
def encode_dataset(csv_path: str, sp_model_path: str) -> list[int]:
    """
    Read again CSV, tokenize with the regex and return a flat list of integers (ID of SentencePiece vocab).
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
        tokens = process_one_song(lyrics)          
        raw = " ".join(tokens)                     
        ids = sp.EncodeAsIds(raw)                  
        all_ids.extend(ids)
    return all_ids


if __name__ == "__main__":
    csv_path   = CSV_PATH
    corpus_path = CORPUS_PATH
    model_prefix = SPM_MODEL_PREFIX
    vocab_sz   = VOCAB_SIZE

    write_corpus(csv_path, corpus_path)

    spm_model = train_spm(
        corpus_path=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_sz,
        character_coverage=CHAR_COVERAGE,
    )

    ids = encode_dataset(csv_path, spm_model)

    # save ids
    ids_path = Path("FreestyleAI/data/ids_spm.pkl")
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    with ids_path.open("wb") as f:
        pickle.dump(ids, f)
    print(f"✅  Lista di ID salvata in: {ids_path}")

    Path(corpus_path).unlink(missing_ok=True)
    print("🗑️  Corpus temporaneo cancellato.")