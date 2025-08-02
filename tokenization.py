import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
db = pd.read_csv('updated_rappers.csv', usecols=["song", "lyric"])

db["lyric"] = db["lyric"].apply(lambda x: x.encode("utf-8"))
songs_names = db['song'].drop_duplicates().tolist()

first3Songs = songs_names[:5]
filtered_db = db[db['song'].isin(first3Songs)]

song_lyrics_bytes = filtered_db.groupby('song')['lyric'].apply(lambda lyrics: b'\n'.join(lyrics)).to_dict()
song_lyrics_indices = {song: list(lyrics_bytes) for song, lyrics_bytes in song_lyrics_bytes.items()}

max_index = max(max(indices) for indices in song_lyrics_indices.values()) #122

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

all_indices = [b for indices in song_lyrics_indices.values() for b in indices]

vocab_size = 150
num_merges = vocab_size - max_index
ids = list(all_indices)

merges = {} # {(pair) : newToken}

for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = max_index + 1 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

print("tokens length:", len(all_indices))
print("ids length:", len(ids))
print(f"compression ratio: {len(all_indices) / len(ids):.2f}X")
