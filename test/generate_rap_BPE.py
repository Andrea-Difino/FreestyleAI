import torch
import torch.nn.functional as F
from tqdm import tqdm
import sentencepiece as spm
from FreestyleAI import WordGramModel # type: ignore


DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPM_MODEL_PATH  = "FreestyleAI/models/bpe_spm_updated.model"   
MODEL_STATE_PATH = "FreestyleAI/models/bpe-model-finetuned.pt"    
MAX_TOKENS      = 150      # lunghezza massima della sequenza generata
TEMPERATURE     = 1.3     # temperatura di base (più alta → più casuale)
TOP_K           = 20       # 0 = disabled, altrimenti tiene i K token più probabili
TOP_P           = 0.9     # nucleus sampling (0 = disabled)
BLOCK_SIZE      = 32      

sp = spm.SentencePieceProcessor()
sp.Load(SPM_MODEL_PATH)

VOCAB_SIZE   = sp.GetPieceSize()
START_ID = sp.PieceToId("<START>")
END_ID   = sp.PieceToId("<END>")
PAD_ID = sp.pad_id()
LINE_ID      = sp.PieceToId("<LINE>")  # se il tuo modello lo usa

print(f"🔠  Vocabulary size: {VOCAB_SIZE}   START={START_ID}  END={END_ID}  LINE={LINE_ID} PAD={PAD_ID}")

model = WordGramModel(VOCAB_SIZE)
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location="cuda"))
model.to(DEVICE)
model.eval()

print(f"✅  Model loaded – parametri totali: {sum(p.numel() for p in model.parameters()):,}")


def sample_from_probs(probs: torch.Tensor, temperature: float, top_k: int = 0, top_p: float = 0.0):
    """
    - `probs`  : (1, vocab)   già softmax
    - `temperature` : scala le log‑probabilità.
    - `top_k`  : tieni i k token più probabili (0 = disabled).
    - `top_p`  : nucleus sampling (mantieni la massa cumulativa ≤ p).
    """
    # temperatura
    if temperature != 1.0:
        logits = torch.log(probs + 1e-9) / temperature
        probs = torch.softmax(logits, dim=-1)

    # top‑k
    if top_k > 0:
        topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)
        probs = probs / probs.sum(dim=-1, keepdim=True)   # normalizzi di nuovo

    # nucleus (top‑p)
    if top_p > 0.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # maschera tutti i token oltre la soglia p
        mask = cumulative > top_p
        # sposta la maschera di una posizione (mantieni il primo token che supera p)
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        # riporta nella posizione originale
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # campiona un token
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id


# ----------------------------------------------------------------------
#   4️⃣  Funzione di generazione
# ----------------------------------------------------------------------
def generate_text(model, sp, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE, top_k: int = TOP_K, top_p: float = TOP_P, block_size: int = BLOCK_SIZE):
    """
    Genera una sequenza di token utilizzando il contesto di `block_size`.
    Restituisce sia la lista di ID che la stringa decodificata.
    """
    model.eval()
    # contesto iniziale = BOS + padding di BOS (così il modello ha sempre block_size token)
    context = [START_ID] + [PAD_ID]*(block_size-1)

    generated_ids = []          # solo i token *generati* (esclude il padding iniziale)
    entropies = []              # opzionale: per analisi
    MIN_TOKENS_BEFORE_STOP = 10

    for step in range(max_tokens):
        # -------------------------------------------------
        #   Forward: prendiamo solo l'ultimo token del blocco
        # -------------------------------------------------
        x = torch.tensor([context], dtype=torch.long, device=DEVICE)   # shape (1, block_size)
        with torch.no_grad():
            logits, _ = model(x) # logits shape (1, block_size, vocab)

        # Consido l'ultimo token della sequenza (indice -1)
        logits_last = logits[:, -1, :] / temperature
        probs = torch.softmax(logits_last, dim=-1)   # (1, vocab)

        # -------------------------------------------------
        #   Entropia 
        # -------------------------------------------------
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).item()
        entropies.append(entropy)

        if entropy < 2.2:
            temperature += 0.2
        elif entropy > 4.0:
            temperature = max(0.5, temperature - 0.1)

        # -------------------------------------------------
        #   Sampling 
        # -------------------------------------------------
        next_id = sample_from_probs(probs, temperature, top_k, top_p)
        generated_ids.append(next_id)

        # -------------------------------------------------
        #   Aggiorna il contesto (shift‑left + nuovo token)
        # -------------------------------------------------
        context = context[1:] + [next_id]

        # -------------------------------------------------
        #   Stop‑condition (se trovi <LINE> o <END>)
        # -------------------------------------------------
        if len(generated_ids) >= MIN_TOKENS_BEFORE_STOP and next_id in {END_ID}:
            break

    # Decodifica in stringa
    text = sp.DecodeIds(generated_ids)
    # Se vuoi trasformare il token <LINE> in un ritorno a capo:
    text = text.replace("<LINE>", "\n")
    text = text.replace("<END>", "")
    # (Puoi anche rimuovere eventuali token di padding residui)
    return generated_ids, text, entropies


# ----------------------------------------------------------------------
#   5️⃣  Esecuzione di prova
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Genera 10 esempi
    for i in range(10):
        ids, txt, ent = generate_text(
            model,
            sp,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            block_size=BLOCK_SIZE,
        )
        print(f"\n🪄  Sample {i+1}")
        print(f"   → IDs   : {ids[:20]} … ({len(ids)} token)")
        print(f"   → Text  : {txt}")
        # media dell'entropia (indicatore di “creatività”)
        if ent:
            print(f"   → Entropy (avg): {sum(ent)/len(ent):.3f}")