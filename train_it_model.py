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

db = pd.read_csv('FreestyleDataset.csv', usecols=["battle_id", "text", "line_number", "dialect"])

battles = list(sorted(set(db["battle_id"])))

battles_text_dict = {title: "" for title in battles}

print(battles_text_dict)

def is_informative(word):
    return (
        not word.isnumeric() and
        not re.search(r'\d', word)
    )

def train():
    context_size = 4
    embedding_dim = 10
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
    print(wotoi)
    print(vocab_size)

    def word_to_index(word):
        return wotoi.get(word, wotoi["<UNK>"])

    def build_dataset():
        X, Y = [], []
        for battle in battles_text_dict.keys():
            context = [wotoi["<START>"]] * context_size
            for word in battles_text_dict[battle].split():
                ix = word_to_index(word)
                X.append(context.copy())
                Y.append(ix)
                context = context[1:] + [ix]
        return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

    Xtr, Ytr = build_dataset()
    print(f"Dimensione Xtr: {Xtr.shape}")
    dataset = TensorDataset(Xtr, Ytr)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    print(train_size)
    print(test_size)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = WordGramModel(vocab_size, embedding_dim, context_size, hidden_dim=512).to(device)
    model.train()

    lossi = []
    min_loss = float('inf')
    counter = 0
    patience = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)

    for epoch in range(20):
        print(f"Epoch {epoch + 1}/{20}")
        loop = tqdm(train_loader, leave=False)
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
        print(f"\nEpoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            #torch.save(model.state_dict(), f"40tanh512hidden_best_model{epoch_loss:.2f}.pt")
            print("Best Model.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    plt.plot(lossi)
    plt.show()
    test_loss(model, test_loader)

    torch.save(model.state_dict(), 'models/ita-word-gram_model.pt')
    metadata = {
        "wotoi": wotoi,
        "itow": itow,
        "context_size": context_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": 512
    }
    torch.save(metadata, 'metadata/ita-word-gram_metadata.pt')

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