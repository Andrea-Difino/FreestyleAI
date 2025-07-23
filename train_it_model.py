import torch.nn.functional as F
import pandas as pd
import torch
from architecture import WordGramModel
from matplotlib import pyplot as plt
import re
from collections import Counter
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

db = pd.read_csv('FreestyleDataset.csv', usecols=["battle_id", "text", "line_number", "dialect"])

batch_size = 4 #sequences to be processed in parallel
block_size = 16 #number of words to be processed in parallel = (context_size)
max_iters = 300 #number of iterations to train
eval_interval = 50 #how many iterations to wait before evaluating the model
learning_rate = 3e-4 #learning rate for the optimizer
eval_iters = 200
n_embd = 32

battles = list(sorted(set(db["battle_id"])))

battles_text_dict = {title: "" for title in battles}

def is_informative(word):
    return (
        not word.isnumeric() and
        not re.search(r'\d', word)
    )

def refine_data():
    for _, row in db.iterrows():
        title = row['battle_id']
        text = row['text']
        dialect = row['dialect']
        if dialect:
            continue
        cleaned_lyrics = ""
        for word in text.split():
            if is_informative(word):
                if word in ["<USER>", "IA"]:
                    cleaned_lyrics += word + " "
                else:
                    cleaned_lyrics += word.lower() + " "
            else:
                cleaned_lyrics += "<UNK> "
        cleaned_lyrics = cleaned_lyrics.strip()
        battles_text_dict[title] += "<START> " + cleaned_lyrics + " <END> "

    print(battles_text_dict)
    all_words = []
    for song in battles_text_dict:
        all_words.extend(battles_text_dict[song].split())

    vocab = sorted(set(word for word in all_words if word not in ["<START>", "<UNK>", "<END>"]))
    
    wotoi = {"<START>": 0, "<UNK>": 1, "<END>": 2}

    for i, word in enumerate(sorted(vocab), start=3):
        wotoi[word] = i

    itow = {i: w for w, i in wotoi.items()}
    vocab_size = len(wotoi)
    return vocab_size, wotoi, itow

vocab_size, wotoi, itow = refine_data()
decode = lambda l : ''.join([itow[i] for i in l])

def encode(words):
    return [wotoi.get(w, wotoi["<UNK>"]) for w in words]

def divide_data():
    data = torch.tensor([], dtype = torch.long)

    for battle in battles_text_dict.keys():
        words = battles_text_dict[battle].split()

        # Converti parole in indici
        word_indices = torch.tensor(encode(words), dtype = torch.long)
        data = torch.cat((data, word_indices), dim = 0)

    split_v = int(0.9*len(data))
    train_data = data[:split_v]
    val_data = data[split_v:]
    return train_data, val_data

train_data, val_data = divide_data()

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
    patience = 10
    epoch_losses = []

    for epoch in range(max_iters):
        print(f"Epoch {epoch + 1}/{max_iters}") 

        xb,yb = get_batch('train')

        logits, loss = model(xb,yb)

        epoch_losses.append(loss)
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            print("Best Model.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    #optimized plot to have a clear view of the loss
    window = 10 
    smoothed = torch.tensor(epoch_losses).view(-1, window).mean(dim=1)
    plt.plot(smoothed)

    losses = estimate_loss()
    performanceLog = f'\nIt-Model\nTraining loss: {losses['train']:.4f} - Val loss : {losses['val']:.4f}\n' + f'Vocab: {vocab_size}W , Architecture: cs{block_size}-ed{n_embd}\n'

    with open('performance_log.txt', 'a') as f:
            f.write(performanceLog)

    torch.save(model.state_dict(), 'models/it-word-gram_model.pt')
    metadata = {
        "wotoi": wotoi,
        "itow": itow,
        "context_size": block_size,
        "embedding_dim": n_embd
    }
    torch.save(metadata, 'metadata/it-word-gram_metadata.pt')

if __name__ == "__main__":
    train()