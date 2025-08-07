import torch
import torch.nn.functional as F
from architecture import WordGramModel
from matplotlib import pyplot as plt
from tqdm import tqdm
from tokenizer import tokenize, decode

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Get BPE tokenized data
vocab, reverse_vocab, merges, ids = tokenize()
vocab_size = len(vocab)
max_index = max(vocab.values())
# Model config
batch_size = 256
block_size = 32
learning_rate = 0.05
eval_iters = 200
n_embd = 256

# Split token list into training and validation

def divide_data():
    split_idx = int(0.8 * len(ids))
    train_data = torch.tensor(ids[:split_idx], dtype=torch.long)
    val_data = torch.tensor(ids[split_idx:], dtype=torch.long)
    return train_data, val_data
 
train_data, val_data = divide_data() #33.127.904
print(train_data.shape)
# Get batch of training data

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# Estimate loss on train and val sets

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

# Initialize model
model = WordGramModel(vocab_size).to(device)
model.train()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)

# Train loop

def train():
    print("Inizio addestramento del modello...")

    min_loss = float('inf')
    counter = 0
    patience = 10
    epoch_losses = []
    steps_per_epoch = 2250
    epochs = 125

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
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    # Plot
    plt.plot(torch.tensor(epoch_losses))
    plt.title("Train loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("performance/BPE_loss_plot.png")
    plt.show()
    plt.close()

    # Save model
    losses = estimate_loss()
    print(losses)
    torch.save(model.state_dict(), 'models/bpe-model.pt')
    torch.save({
        "merges": merges,
        "max_index": max_index,
        "context_size": block_size,
        "embedding_dim": n_embd,
        "vocab-size": vocab_size, 
        "reverse_vocab": reverse_vocab
    }, 'metadata/bpe-metadata.pt')

    with open('performance/performance_log.txt', 'a') as f:
        f.write(f"\nBPE-Model\nTrain loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}\n")


if __name__ == "__main__":
    train()
