import time, pickle, torch
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    block_size = 256
    eval_iters = 200
    emb_dim    = 384
    learning_rate = 0.0005
    epochs   = 125
    patience = 10
    log_interval = 50  # aggiornamento tensorboard

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
    model = WordGramModel(VOCAB_SIZE, emb_dim, block_size, dropout = 0.25).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # ------------------- TensorBoard Setup -------------------
    run_name = datetime.now().strftime("%Y%m%d_%H%M") + "dropout0.25-wd0.01"

    writer_train = SummaryWriter(f"FreestyleAI/logs/{run_name}/train")
    writer_val   = SummaryWriter(f"FreestyleAI/logs/{run_name}/val")
    best_val_loss = float('inf')
    no_improve = 0
    
    global_step = 0 # Per asse X di TensorBoard continuo tra le epoche

    # ------------------- Training loop -------------------
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in pbar:
            xb, yb = unpack_batch(batch, DEVICE)

            # Forward
            # return_activations=False perché non stiamo plottando istogrammi, risparmiamo memoria
            logits, loss = model(xb, yb, return_activations=False) 

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Logging Batch su TensorBoard
            if global_step % log_interval == 0:
                writer_train.add_scalar("Loss/batch", loss.item(), global_step)
                
                # Monitoraggio Gradient Norm (Leggero sulla RAM)
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer_train.add_scalar("System/GradientNorm", total_norm, global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # ------------------- FINE EPOCA (Validazione e Checkpoint) -------------------
        # Calcoliamo la Loss vera su Train e Validation set
        print(f"⏳ Stima loss fine epoca {epoch+1}...")
        losses = estimate_loss(model, train_loader, val_loader, eval_iters, DEVICE)
        
        print(f"📉 Epoch {epoch+1}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
        
        # Scrivi su TensorBoard
        writer_train.add_scalar("Loss/epoch", losses['train'], epoch)
        writer_val.add_scalar("Loss/epoch", losses['val'], epoch)

        # --- Early Stopping & Checkpoint (Sulla VALIDATION Loss) ---
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            no_improve = 0
            # Salviamo il modello migliore
            torch.save(model.state_dict(), "FreestyleAI/models/bpe-model.pt")
            print(f"💾  Nuovo miglior modello salvato! (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"⚠️  Nessun miglioramento per {no_improve}/{patience} epoche.")
            if no_improve >= patience:
                print("⏹️  Early stopping triggered!")
                break

    writer_train.close()
    writer_val.close()

    # ------------------- Save metadata -------------------
    torch.save({
        "sp_model_path": SPM_MODEL_PATH,
        "context_size": block_size,
        "embedding_dim": emb_dim,
        "vocab_size": VOCAB_SIZE,
    }, "FreestyleAI/metadata/bpe-metadata.pt")

    # ------------------- Timing -------------------
    hrs = (time.time() - start_time) / 3600.0
    print(f"\n⏱️  Tempo totale: {hrs:.2f} ore")

if __name__ == "__main__":
    main()