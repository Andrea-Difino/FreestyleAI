import torch
import time
import torch.nn.functional as F
from FreestyleAI import WordGramModel
from matplotlib import pyplot as plt
from tqdm import tqdm
from .tokenizer import tokenize
from torch.utils.data import TensorDataset, DataLoader


start_time = time.time()
# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Get BPE tokenized data
print("Starting tokenization...")
vocab, reverse_vocab, merges, ids = tokenize()
print("Tokenization finished!")

vocab_size = len(vocab)
max_index = max(vocab.values())
# Model config
batch_size = 256
block_size = 32
eval_iters = 200
n_embd = 256

# Split token list into training and validation

split_idx = int(0.8 * len(ids))
train_ids = torch.tensor(ids[:split_idx], dtype=torch.long, device=device)
val_ids   = torch.tensor(ids[split_idx:], dtype=torch.long, device=device) 

print("Training data shape: ", train_ids.shape)
# Get batch of training data

def make_dataset(tensor):
    # crea sequenze di length=block_size+1 (x + y)
    seq_len = block_size + 1
    # numero di blocchi possibili (non overlapping)
    n_blocks = (len(tensor) - 1) // seq_len
    tensor = tensor[: n_blocks * seq_len]          # troncamento
    tensor = tensor.view(n_blocks, seq_len)        # (N, seq_len)
    return TensorDataset(tensor)

train_dataset = make_dataset(train_ids)
val_dataset   = make_dataset(val_ids)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
    drop_last=True,
)

# -------------------------------------------------
# Helper per ottenere x, y (sul device)
# -------------------------------------------------
def unpack_batch(batch):
    seq = batch[0]                     # (B, block+1)
    xb = seq[:, :-1].contiguous()      # (B, block)
    yb = seq[:, 1:].contiguous()       # (B, block)
    return xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

# -------------------------------------------------
# Stima della loss (valutazione)
# -------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for i, batch in enumerate(loader):
            if i >= eval_iters:
                break
            xb, yb = unpack_batch(batch)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# Initialize model
model = WordGramModel(vocab_size).to(device)
model.train()
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr = 0.05, 
    momentum=0.9
)

# Train loop

def train():
    print("Inizio addestramento del modello...")

    min_loss = float('inf')
    counter = 0
    patience = 10

    epoch_losses = []
    epochs = 125

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False)

        for batch in pbar:
            xb, yb = unpack_batch(batch)

            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Avg train loss (epoch {epoch+1}): {avg_loss:.4f}")

        # Early‑stopping
        if avg_loss < min_loss:
            min_loss = avg_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    # -------------------------------------------------
    # Plot dei loss per epoca
    # -------------------------------------------------
    plt.plot(torch.tensor(epoch_losses))
    plt.title("Train loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("FreestyleAI/performance/BPE_loss_plot.png")
    plt.close()

    # Save model
    losses = estimate_loss()
    print(losses)
    torch.save(model.state_dict(), 'FreestyleAI/models/bpe-model.pt')
    torch.save({
        "merges": merges,
        "max_index": max_index,
        "context_size": block_size,
        "embedding_dim": n_embd,
        "vocab-size": vocab_size, 
        "reverse_vocab": reverse_vocab
    }, 'FreestyleAI/metadata/bpe-metadata.pt')
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Tempo di esecuzione: {duration/120} ore")

    with open('FreestyleAI/performance/performance_log.txt', 'a') as f:
        f.write(
            f"\nBPE-Model\nTrain loss: {losses['train']:.4f}, " 
            f"Val loss: {losses['val']:.4f}, Tempo di esecuzione in ore: {duration/120}\n"
        )

    
if __name__ == "__main__":
    train()
