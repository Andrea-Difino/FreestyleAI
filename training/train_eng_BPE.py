import time, pickle, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import numpy as np
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
    batch_size = 64
    block_size = 64
    eval_iters = 200
    emb_dim    = 384
    learning_rate = 0.0005
    epochs   = 125
    patience = 10
    plot_interval = 64  # aggiornamento grafico ogni N batch

    # ------------------- Train / Val split -------------------
    split_idx = int(0.8 * len(ids_int))
    train_ids = torch.tensor(ids_int[:split_idx], dtype=torch.long)
    val_ids   = torch.tensor(ids_int[split_idx:], dtype=torch.long)

    train_dataset = make_dataset(train_ids, block_size)
    val_dataset   = make_dataset(val_ids, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=True)

    # ------------------- Model & Optimizer -------------------
    model = WordGramModel(VOCAB_SIZE, emb_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------- Setup live plot -------------------
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    axs = axs.flatten()
    losses_plot = []
    line_loss, = axs[0].plot([], [], label="Train Loss")
    axs[0].set_title("Train loss per batch")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # ------------------- Training loop -------------------
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch_idx, batch in enumerate(pbar, 1):
            xb, yb = unpack_batch(batch, DEVICE)

            # forward con attivazioni
            logits, loss, activations = model(xb, yb, return_activations=True)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            losses_plot.append(loss.item())
            pbar.set_postfix(loss=loss.item())

            # --- Aggiorna plot solo ogni N batch ---
            if batch_idx % plot_interval == 0:
                # Loss
                line_loss.set_xdata(range(len(losses_plot)))
                line_loss.set_ydata(losses_plot)
                axs[0].relim(); axs[0].autoscale_view()

                # Gradienti
                axs[1].cla()
                all_grads = [p.grad.view(-1).cpu().numpy() for p in model.parameters() if p.grad is not None]
                if all_grads:
                    all_grads = np.concatenate(all_grads)
                    axs[1].hist(all_grads, bins=20, color="blue", alpha=0.7)
                    axs[1].set_title("Distribuzione gradienti")

                # Attivazioni neuroni
                axs[2].cla()
                mean_acts = [act.abs().mean().item() for act in activations]
                axs[2].bar(range(len(mean_acts)), mean_acts, color="orange", alpha=0.7)
                axs[2].set_title("Attivazioni medie dei neuroni")
                axs[2].set_xlabel("Layer")
                axs[2].set_ylabel("Mean abs activation")

                plt.draw()
                plt.pause(0.01)

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"\n📈  Avg train loss epoch {epoch+1}: {avg_epoch_loss:.4f}")

        # Early-stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹️  Early stopping triggered")
                break

    plt.ioff()
    fig.savefig("FreestyleAI/performance/BPE_training_plots.png")
    plt.close()

    # ------------------- Final evaluation -------------------
    final_losses = estimate_loss(model, train_loader, val_loader, eval_iters, DEVICE)
    print("\n🔎  Final loss:", final_losses)

    # ------------------- Save model + metadata -------------------
    torch.save(model.state_dict(), "FreestyleAI/models/bpe-model.pt")
    torch.save({
        "sp_model_path": SPM_MODEL_PATH,
        "context_size": block_size,
        "embedding_dim": emb_dim,
        "vocab_size": VOCAB_SIZE,
    }, "FreestyleAI/metadata/bpe-metadata.pt")

    # ------------------- Timing -------------------
    hrs = (time.time() - start_time) / 3600.0
    print(f"\n⏱️  Tempo di esecuzione: {hrs:.2f} ore")
    with open("FreestyleAI/performance/performance_log.txt", "a") as f:
        f.write(
            f"\nBPE_SPM-Model - Train loss: {final_losses['train']:.4f}, "
            f"Val loss: {final_losses['val']:.4f}, Tempo (h): {hrs:.2f}\n"
        )

if __name__ == "__main__":
    main()