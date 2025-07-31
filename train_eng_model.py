import torch.nn.functional as F
import pandas as pd
import torch
from architecture import WordGramModel
from matplotlib import pyplot as plt
import re
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import Counter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
db = pd.read_csv('updated_rappers.csv', usecols=["song", "lyric"])
db["lyric"] = db["lyric"].apply(lambda x: x.lower())
songs_names = list(dict.fromkeys(db['song']))

song_lyrics_dict = {title: "" for title in songs_names}

batch_size = 64 #sequences to be processed in parallel
block_size = 32 #number of words to be processed in parallel = (context_size)
max_iters = 8000 #number of iterations to train
eval_interval = 50 #how many iterations to wait before evaluating the model
learning_rate = 1e-4 #learning rate for the optimizer
eval_iters = 200
n_embd = 256


def is_informative(word):
    return (
        not word.isnumeric() and
        not re.search(r'\d', word) and
        not re.fullmatch(r'\W+', word) and
        re.search('[a-zA-Z]', word) and
        not re.search(r'[aeiou]{3,}', word.lower()) and
        not re.search(r'(.)\1{2,}', word.lower())
    )

def clean_text(text):
    # Adds space around common punctuation: . , ! ? ( )
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    # Replaces multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def refine_data():
    for _, row in db.iterrows():
        title = row['song']
        lyrics = row['lyric']
        cleaned_lyrics = "<START> "
        lyrics = clean_text(lyrics)
        for word in lyrics.split():
            if is_informative(word):
                cleaned_lyrics += word + " "
            else:
                cleaned_lyrics += "<UNK> "
        cleaned_lyrics = cleaned_lyrics.strip()
        song_lyrics_dict[title] += cleaned_lyrics + " <END> "

    all_words = []
    for song in song_lyrics_dict:
        all_words.extend(song_lyrics_dict[song].split())

    word_freq = Counter(all_words)
    min_freq = 5

    vocab = [word for word, freq in word_freq.items() if freq >= min_freq and word not in ["<START>","<UNK>","<END>"]]
    # Word to index mapping
    wotoi = {"<START>": 0, "<UNK>": 1, "<END>": 2}

    for i, word in enumerate(sorted(vocab), start=3):
        wotoi[word] = i

    # Index to word mapping
    itow = {i: w for w, i in wotoi.items()}
    vocab_size = len(wotoi)

    return vocab_size, wotoi, itow

vocab_size, wotoi, itow = refine_data()
decode = lambda l : ' '.join([itow[i] for i in l])

def encode(words):
    return [wotoi.get(w, wotoi["<UNK>"]) for w in words]

def divide_data():
    data = torch.tensor([], dtype = torch.long)

    for song in song_lyrics_dict.keys():
        words = song_lyrics_dict[song].split()

        # Converti parole in indici
        word_indices = torch.tensor(encode(words), dtype = torch.long)
        data = torch.cat((data, word_indices), dim = 0)

    split_v = int(0.9*len(data))
    train_data = data[:split_v]
    val_data = data[split_v:]
    return train_data, val_data

train_data, val_data = divide_data()
print(len(train_data))

def get_batch(split): 
    #generate batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device),y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = WordGramModel(vocab_size).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

def train():
    print("Inizio addestramento del modello...")

    min_loss = float('inf')
    counter = 0
    patience = 100
    epoch_losses = []
    steps_per_epoch = 2000
    epochs = 30
 
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}") 
        total_train_loss = 0

        pbar = tqdm(range(steps_per_epoch), desc="Training", leave=False)

        for _ in pbar:
            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / steps_per_epoch
        epoch_losses.append(avg_train_loss)

        print(f"Avg train loss: {avg_train_loss:.4f}")

        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            counter = 0
            print("Best Model.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    plt.plot(torch.tensor(epoch_losses))
    plt.title("Train loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("performances/loss_plot.png")
    plt.show()
    plt.close()

    losses = estimate_loss()
    print(losses)
    # Save performance log
    performanceLog = f'\nEng-Model\nTraining loss: {losses['train']:.4f} - Val loss : {losses['val']:.4f}\n' + f'Vocab: {vocab_size}W , Architecture: cs{block_size}-ed{n_embd}\n'

    with open('performance/performance_log.txt', 'a') as f:
        f.write(performanceLog)

    torch.save(model.state_dict(), 'models/eng-word-gram_model.pt')
    metadata = {
        "wotoi": wotoi,
        "itow": itow,
        "context_size": block_size,
        "embedding_dim": n_embd
    }
    torch.save(metadata, 'metadata/eng-word-gram_metadata.pt')

if __name__ == "__main__":
    train()
