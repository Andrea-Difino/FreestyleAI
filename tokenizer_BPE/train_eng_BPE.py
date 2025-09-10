import time, pickle, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sentencepiece as spm
from FreestyleAI import WordGramModel # type: ignore

def make_dataset(tensor, block_size):
    """Create sequence of length block_size+1 (x + y) without overlap."""
    seq_len = block_size + 1
    n_blocks = (len(tensor) - 1) // seq_len
    tensor = tensor[: n_blocks * seq_len].view(n_blocks, seq_len)
    return TensorDataset(tensor)

def unpack_batch(batch, device):
    seq = batch[0]                     # (B, block+1)  → CPU
    xb = seq[:, :-1].contiguous()
    yb = seq[:, 1:].contiguous()
    return xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    model.eval()
    out = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for i, batch in enumerate(loader):
            if i >= eval_iters:
                break
            xb, yb = unpack_batch(batch, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

def main():
    start_time = time.time()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀  Device:", DEVICE)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    # ------------------- Tokenizer -------------------
    SPM_MODEL_PATH = "FreestyleAI/models/bpe_spm.model"
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL_PATH)
    VOCAB_SIZE = sp.GetPieceSize()
    print(f"🔠  Vocabulary size (SentencePiece): {VOCAB_SIZE}")

    # ------------------- Load pre‑encoded IDs -------------------
    IDS_PATH = Path("FreestyleAI/data/ids_spm.pkl")
    with IDS_PATH.open("rb") as f:
        ids_int = pickle.load(f)         

    print(f"📦  Numero di token (lunghezza del corpus): {len(ids_int):,}")

    # ------------------- Hyper‑params -------------------
    batch_size = 256
    block_size = 32
    eval_iters = 200
    n_embd     = 256
    learning_rate = 0.05
    momentum = 0.9
    epochs   = 125
    patience = 10

    # ------------------- Train / Val split (CPU tensors) -------------------
    split_idx = int(0.8 * len(ids_int))
    train_ids = torch.tensor(ids_int[:split_idx], dtype=torch.long)   # CPU
    val_ids   = torch.tensor(ids_int[split_idx:], dtype=torch.long)   # CPU

    train_dataset = make_dataset(train_ids, block_size)
    val_dataset   = make_dataset(val_ids, block_size)

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

    # ------------------- Model & Optimizer -------------------
    model = WordGramModel(VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
    )

    # ------------------- Training loop -------------------
    print("\n🛠️  Inizio addestramento del modello...")
    best_loss = float('inf')
    no_improve = 0
    epoch_losses = []

    for epoch in range(epochs):
        total = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{epochs}",
                    leave=False)

        for batch in pbar:
            xb, yb = unpack_batch(batch, DEVICE)
            _, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg = total / len(train_loader)
        epoch_losses.append(avg)
        print(f"\n📈  Avg train loss epoch {epoch+1}: {avg:.4f}")

        # Early‑stopping
        if avg < best_loss:
            best_loss = avg
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹️  Early stopping triggered")
                break

    # ------------------- Plot loss -------------------
    plt.figure()
    plt.plot(epoch_losses, marker='o')
    plt.title("Train loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("FreestyleAI/performance/BPE_loss_plot.png")
    plt.close()

    # ------------------- Final evaluation -------------------
    final_losses = estimate_loss(model, train_loader, val_loader, eval_iters, DEVICE)
    print("\n🔎  Final loss:", final_losses)

    # ------------------- Save model + minimal metadata -------------------
    torch.save(model.state_dict(), "FreestyleAI/models/bpe-model.pt")
    torch.save({
        "sp_model_path": SPM_MODEL_PATH,
        "context_size": block_size,
        "embedding_dim": n_embd,
        "vocab_size": VOCAB_SIZE,
    }, "FreestyleAI/metadata/bpe-metadata.pt")

    # ------------------- Timing -------------------
    hrs = (time.time() - start_time) / 3600.0   # ore, non 120!
    print(f"\n⏱️  Tempo di esecuzione: {hrs:.2f} ore")
    with open("FreestyleAI/performance/performance_log.txt", "a") as f:
        f.write(
            f"\nBPE_SPM-Model - Train loss: {final_losses['train']:.4f}, "
            f"Val loss: {final_losses['val']:.4f}, Tempo (h): {hrs:.2f}\n"
        )

if __name__ == "__main__":
    main()