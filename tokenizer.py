import pandas as pd
import torch

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def tokenize():
    db = pd.read_csv('updated_rappers.csv', usecols=["song", "lyric"])

    db["lyric"] = db["lyric"].apply(lambda x: x.encode("utf-8"))
    songs_names = db['song'].drop_duplicates().tolist()

    filtered_db = db[db['song'].isin(songs_names)]

    full_corpus = b'\n'.join(filtered_db['lyric'])
    ids = list(full_corpus)

    max_index = max(ids) #122

    vocab_size = max_index + 30
    num_merges = vocab_size - max_index - 1

    assert max(ids) < vocab_size, "Token fuori dal vocabolario!"

    merges = {} # {(pair) : newToken}

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = max_index + 1 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    print(f"compression ratio: {len(list(full_corpus)) / len(ids):.2f}X")

    #list(full_corpus) = indices of the original bytes 

    #ids = list of indices after the compression

    return merges, max_index, list(full_corpus), ids , vocab_size

def encode(text, merges): 
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2: 
        stats = get_stats(tokens)
        pair = min(stats, key = lambda p: merges.get(p, float("inf")))
        if pair not in merges: 
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def build_vocab(merges, max_index):
    vocab = {i: bytes([i]) for i in range(max_index)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

def decode(ids, vocab):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


if __name__ == "__main__":
    # esegui solo se lanci direttamente questo file
    merges, max_index, all_indices, ids, vocab_size = tokenize()