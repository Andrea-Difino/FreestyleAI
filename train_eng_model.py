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
print("Device in uso:", device)

print("CUDA disponibile:", torch.cuda.is_available())
print("GPU in uso:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nessuna")

db = pd.read_csv('updated_rappers.csv', usecols=["song", "lyric"])
db["lyric"] = db["lyric"].apply(lambda x: x.lower())
songs_names = list(dict.fromkeys(db['song']))

song_lyrics_dict = {title: "" for title in songs_names}

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
    text = re.sub(r'([.,!?()])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'([a-zA-Z])[,]([a-zA-Z])', r'\1 , \2', text)
    return text.strip()

def refine_data():

    for _, row in db.iterrows():
        title = row['song']
        lyrics = row['lyric']
        cleaned_lyrics = "<START> "
        clean_text(lyrics)
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
    wotoi = {"<START>": 0, "<UNK>": 1, "<END>": 2}
    for i, word in enumerate(sorted(vocab), start=3):
        wotoi[word] = i
    itow = {i: w for w, i in wotoi.items()}
    vocab_size = len(wotoi)

    return vocab_size, wotoi, itow

def n_word_gram(word_to_index, context_size):
    X, Y = [], []
    for song in song_lyrics_dict.keys():
        context = [word_to_index["<START>"]] * context_size
        for word in song_lyrics_dict[song].split():
            ix = word_to_index.get(word, word_to_index["<UNK>"])
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

def train():
    context_size = 4
    embedding_dim = 40

    vocab_size, wotoi, itow = refine_data()

    Xtr, Ytr = n_word_gram(wotoi, context_size)

    print(f"Dimensione Xtr: {Xtr.shape}")
    dataset = TensorDataset(Xtr, Ytr)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = WordGramModel(vocab_size, embedding_dim, context_size, hidden_dim=512).to(device)
    model.train()

    lossi = []
    min_loss = float('inf')
    counter = 0
    patience = 3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009)

    for epoch in range(10):
        print(f"Epoch {epoch + 1}/{10}")
        loop = tqdm(train_loader, leave=True)
        epoch_losses = []
        for x_batch, y_batch in loop:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossi.append(loss.item())
            epoch_losses.append(loss.item())

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            print("Best Model.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    plt.plot(lossi)
    plt.show()
    test_loss(model, test_loader)

    torch.save(model.state_dict(), 'models/eng-word-gram_model.pt')
    metadata = {
        "wotoi": wotoi,
        "itow": itow,
        "context_size": context_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": 512
    }
    torch.save(metadata, 'metadata/eng-word-gram_metadata.pt')

    pass

def test_loss(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")


if __name__ == "__main__":
    train()
