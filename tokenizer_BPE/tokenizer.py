import pandas as pd
import regex as re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<START>", "<END>", "<UNK>", "<LINE>"]

def is_informative(word):
    word_stripped = word.strip()
    if not word_stripped:
        return False
    if word_stripped.lower() in {"<unk>", "<line>", "<start>", "<end>"}:
        return False
    if word_stripped.isdigit():
        return True
    if re.match(r"^[a-zA-Z']+$", word_stripped):
        return True
    if word == " ":
        return True
    return False

def clean_text(text):
    # Replaces multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def refine_data(db):
    all_tokens = []

    # Raggruppa per canzone
    grouped = db.groupby("song")["lyric"].apply(list)

    for _, lyrics_lines in grouped.items():
        song_tokens = ["<START>"]

        for line in lyrics_lines:
            line = clean_text(line.lower()).strip()
            if not line:
                continue
            
            for word in re.findall(GPT4_SPLIT_PATTERN, line):
                
                if is_informative(word):
                    song_tokens.append(word)
                else:
                    song_tokens.append("<UNK>")

            song_tokens.append("<LINE>")  # fine riga

        # Rimuovi ultimo <LINE> e aggiungi <END>
        if song_tokens[-1] == "<LINE>":
            song_tokens[-1] = "<END>"
        else:
            song_tokens.append("<END>")

        all_tokens.append(song_tokens)

    # Flatten
    flat_tokens = []
    for song_tokens in all_tokens:
        flat_tokens.extend(song_tokens)

    return flat_tokens


def tokens_to_bytes(tokens):
    """
    Convert tokens list into list of byte IDs + SEP as separator.
    Special tokens (<START>, <END>, <UNK>) stay as strings, others become bytes + SEP.
    """
    out = []
    for t in tokens:
        if t in SPECIAL_TOKENS:
            out.append(t)  # keep special tokens as is
        else:
            b = t.encode("utf-8")
            out.extend(b)
    return out

def get_stats(ids, reverse_vocab):
    pairs = defaultdict(int)
    for i in range(len(ids) - 1):
        left = reverse_vocab[ids[i]]
        right = reverse_vocab[ids[i+1]]
        if isinstance(left, str) or isinstance(right, str):
            continue 
        pair = (ids[i], ids[i+1])
        pairs[pair] += 1
    return Counter(pairs)

def merge(ids, pair, new_token):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(new_token)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def tokenize():
    db = pd.read_csv(
        "FreestyleAI/updated_rappers.csv",
        usecols=["song", "lyric"],
        engine="pyarrow",
    )

    tokens = refine_data(db)  # regex tokenize + clean + add special tokens
    ids = tokens_to_bytes(tokens)  # converti in bytes + sep (special tokens rimangono)
    # Costruisci vocabolario iniziale: special tokens + byte 0-255 + SEP
    vocab = {}
    reverse_vocab = {}
    current_index = 0

    # Special tokens
    for st in SPECIAL_TOKENS:
        vocab[st] = current_index
        reverse_vocab[current_index] = st
        current_index += 1

    # byte tokens 0-255
    for i in range(256):
        b = i
        vocab[b] = current_index
        reverse_vocab[current_index] = b
        current_index += 1

    # Trasforma ids (token speciali stringa + int byte) in lista di ID interi
    ids_int = []
    for x in ids:
        if isinstance(x, str):
            ids_int.append(vocab[x])  # special token
        else:
            ids_int.append(vocab[x])  # byte (int)

    vocab_size = current_index + 901
    num_merges = vocab_size - current_index - 1

    merges = {}  # {(pair): new_token_id}

    for i in range(num_merges):
        print("merging status: " + str(i+1) + f'/{num_merges-1}')
        stats = get_stats(ids_int, reverse_vocab)
        if not stats:
            break
        most_common = stats.most_common(1)[0][0]

        # Crea nuovo token concatenando i due token in bytes (o string se special token)
        left = reverse_vocab[most_common[0]]
        right = reverse_vocab[most_common[1]]
        print(f"Merge {i+1}: {most_common} → token {current_index}")
        

        if isinstance(left, int):
            if left <= 255:
                left_bytes = bytes([left])  # singolo byte
            else:
                left_bytes = reverse_vocab[left]  # prendi i bytes concatenati dal vocabolario inverso
        else:
            left_bytes = left

        if isinstance(right, int):
            if right <= 255:
                right_bytes = bytes([right])
            else:
                right_bytes = reverse_vocab[right]
        else:
            right_bytes = right

        new_token_bytes = left_bytes + right_bytes
        if new_token_bytes in vocab:
            continue  
        print(f"→ left_bytes: {left_bytes}, right_bytes: {right_bytes}, merged: {new_token_bytes}")
        vocab[new_token_bytes] = current_index
        reverse_vocab[current_index] = new_token_bytes
        merges[most_common] = current_index
        current_index += 1

        prev_length = len(ids_int)
        ids_int = merge(ids_int, most_common, vocab[new_token_bytes])
        new_length = len(ids_int)

        delta = prev_length - new_length
        print(f"Dopo merge: length={new_length}, delta={delta}")

        if delta < 10:
            print("Merge marginali, interruzione anticipata.")
            break

    print(f"Compressione: {len(ids)} → {len(ids_int)} , {(len(ids)/len(ids_int)):.2f}x")    
    
    return vocab, reverse_vocab, merges, ids_int

def encode(text, merges, vocab):
    tokens = re.findall(GPT4_SPLIT_PATTERN, text.lower())
    # sostituisci token non informativi con <UNK>
    tokens = [t if is_informative(t) else "<UNK>" for t in tokens]
    # aggiungi start/end
    tokens = ["<START>"] + tokens + ["<END>"]

    # converti in byte + sep
    ids = tokens_to_bytes(tokens)

    # converto in id interi
    ids_int = []
    for x in ids:
        if isinstance(x, str):
            ids_int.append(vocab[x])
        else:
            ids_int.append(vocab[x])

    # applica merges BPE
    already_merged = set()

    while True:
        stats = get_stats(ids_int, reverse_vocab)
        pair = None
        for p in stats:
            if p in merges and p not in already_merged:
                pair = p
                already_merged.add(p)
                break
        if not pair:
            break
        ids_int = merge(ids_int, pair, merges[pair])

    return ids_int

def decode(ids_int, reverse_vocab):
    bytes_list = []
    for idx in ids_int:
        token = reverse_vocab[idx]
        if isinstance(token, str):
            # Gestione dei token speciali
            if token == "<LINE>":
                bytes_list.append(b"\n")
            elif token in ("<START>", "<END>", "<UNK>"):
                continue 
        else:
            # token bytes (singolo o concatenato)
            if isinstance(token, int):
                bytes_list.append(bytes([token]))
            else:
                bytes_list.append(token)

    decoded = b''.join(bytes_list)
    
    # Pulizia finale: rimuove token speciali se non già gestiti
    decoded = decoded.strip()
 
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return decoded.decode("utf-8", errors="replace")

if __name__ == "__main__":
    vocab, reverse_vocab, merges, ids = tokenize()

    test_text = "I'm feeling good, yeah!"
    encoded = encode(test_text, merges, vocab)
    print("Encoded:", encoded)
    decoded = decode(encoded, reverse_vocab)
    print("Decoded:", decoded)