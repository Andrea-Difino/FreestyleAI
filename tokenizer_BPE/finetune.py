import time, pickle, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import sentencepiece as spm
from FreestyleAI import WordGramModel  # type: ignore

matplotlib.use("Agg")

def make_dataset(tensor, block_size):
    seq_len = block_size + 1
    n_blocks = (len(tensor) - 1) // seq_len
    tensor = tensor[: n_blocks * seq_len].view(n_blocks, seq_len)
    return TensorDataset(tensor)

def unpack_batch(batch, device):
    seq = batch[0]
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

    # ------------------- Tokenizer aggiornato -------------------
    SPM_MODEL_PATH = "FreestyleAI/models/bpe_spm_updated.model"  # nuovo tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL_PATH)
    VOCAB_SIZE = sp.GetPieceSize()
    print(f"🔠  Vocabulary size (SentencePiece): {VOCAB_SIZE}")

    # ------------------- Carica IDs aggiornati -------------------
    IDS_PATH = Path("FreestyleAI/data/ids_spm_updated.pkl")
    with IDS_PATH.open("rb") as f:
        ids_int = pickle.load(f)

    print(f"📦  Numero di token (lunghezza del corpus): {len(ids_int):,}")

    # ------------------- Hyper-params (puoi abbassare LR per FT) -------------------
    batch_size = 256
    block_size = 32
    eval_iters = 200
    n_embd     = 256
    learning_rate = 0.005   
    momentum = 0.9
    epochs   = 30           # di solito meno epoche per FT
    patience = 5

    # ------------------- Train / Val split -------------------
    split_idx = int(0.8 * len(ids_int))
    train_ids = torch.tensor(ids_int[:split_idx], dtype=torch.long)
    val_ids   = torch.tensor(ids_int[split_idx:], dtype=torch.long)

    train_dataset = make_dataset(train_ids, block_size)
    val_dataset   = make_dataset(val_ids, block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=4, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2, drop_last=True,
    )

    # ------------------- Modello aggiornato -------------------
    model = WordGramModel(VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load("FreestyleAI/models/bpe-model-updated.pt", map_location=DEVICE)) 

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum,
    )

    # ------------------- Training loop -------------------
    print("\n🛠️  Inizio fine-tuning del modello...")
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

        # Early-stopping
        if avg < best_loss:
            best_loss = avg
            no_improve = 0
            torch.save(model.state_dict(), "FreestyleAI/models/bpe-model-finetuned.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹️  Early stopping triggered")
                break

    # ------------------- Plot loss -------------------
    plt.figure()
    plt.plot(epoch_losses, marker='o')
    plt.title("Train loss per epoch (fine-tuning)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("FreestyleAI/performance/BPE_loss_plot_finetune.png")
    plt.close()

    # ------------------- Final evaluation -------------------
    final_losses = estimate_loss(model, train_loader, val_loader, eval_iters, DEVICE)
    print("\n🔎  Final loss:", final_losses)

    # ------------------- Metadata -------------------
    torch.save({
        "sp_model_path": SPM_MODEL_PATH,
        "context_size": block_size,
        "embedding_dim": n_embd,
        "vocab_size": VOCAB_SIZE,
    }, "FreestyleAI/metadata/bpe-metadata-finetuned.pt")

    # ------------------- Timing -------------------
    hrs = (time.time() - start_time) / 3600.0
    print(f"\n⏱️  Tempo di esecuzione: {hrs:.2f} ore")

if __name__ == "__main__":
    main()
