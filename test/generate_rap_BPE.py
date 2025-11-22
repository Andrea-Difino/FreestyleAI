import torch
import torch.nn.functional as F
from tqdm import tqdm
import sentencepiece as spm
from FreestyleAI import WordGramModel # type: ignore


DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPM_MODEL_PATH  = "FreestyleAI/models/bpe_spm.model"   
MODEL_STATE_PATH = "FreestyleAI/models/bpe-model.pt"    
MAX_TOKENS      = 150      # lunghezza massima della sequenza generata
TEMPERATURE     = 0.9     # temperatura di base (più alta → più casuale)
TOP_P           = 0.9     # nucleus sampling (0 = disabled)
BLOCK_SIZE      = 64      

sp = spm.SentencePieceProcessor()
sp.Load(SPM_MODEL_PATH)

VOCAB_SIZE = sp.GetPieceSize()
START_ID   = sp.PieceToId("<START>")
END_ID     = sp.PieceToId("<END>")
PAD_ID     = sp.pad_id()
LINE_ID    = sp.PieceToId("<LINE>")  # se il tuo modello lo usa

print(f"🔠  Vocabulary size: {VOCAB_SIZE}   START={START_ID}  END={END_ID}  LINE={LINE_ID} PAD={PAD_ID}")

model = WordGramModel(VOCAB_SIZE, 384)
model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location="cuda"))
model.to(DEVICE)
model.eval()

print(f"✅  Model loaded – parametri totali: {sum(p.numel() for p in model.parameters()):,}")

def apply_repetition_penalty(logits, sequence, penalty=1.2):
    """
    Penalizza i token già presenti nella 'sequence'.
    penalty > 1.0 riduce la probabilità.
    """
    # Prendi l'insieme unico dei token generati (o solo gli ultimi N)
    unique_tokens = set(sequence) 
    
    for token_id in unique_tokens:
        # Se il logit è negativo, moltiplichiamo per rendere il numero più piccolo (più negativo)
        # Se positivo, dividiamo per renderlo più piccolo (meno positivo)
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
            
    return logits

def sample_from_probs(probs: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    Sampling pulito senza ricalcolo della temperatura.
    """
    # Top-K
    if top_k > 0:
        v, _ = torch.topk(probs, top_k)
        # Maschera tutto ciò che è sotto il k-esimo valore
        probs[probs < v[:, [-1]]] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # Nucleus (Top-P)
    if top_p > 0.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        
        # Riporta all'ordine originale
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # Campionamento
    next_id = torch.multinomial(probs, num_samples=1).item()
    return next_id


# ----------------------------------------------------------------------
#   4️⃣  Funzione di generazione
# ----------------------------------------------------------------------
def generate_text(model, sp, start_prompt: str = "", max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE, top_p: float = TOP_P, block_size: int = BLOCK_SIZE):
    
    model.eval()

    context = [START_ID]
    if start_prompt:
        context += sp.EncodeAsIds(start_prompt)

    generated_ids = []
    entropies = []
    MIN_TOKENS_BEFORE_STOP = 10

    for step in range(max_tokens):
        # Prepara l'input. Se il modello richiede una lunghezza fissa, 
        # qui dovresti gestire il padding, ma i Transformer solitamente gestiscono
        # sequenze variabili fino al block_size.
        
        # Tagliamo il contesto se supera il block_size (Sliding Window)
        if len(context) > block_size:
            context = context[-block_size:]

        x = torch.tensor([context], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits, _ = model(x) 
        
        # Prendi i logits dell'ultimo token
        logits_last = logits[:, -1, :]
        
        # Evita di generare START o PAD o UNK se necessario
        logits_last[:, START_ID] = float('-inf')
        logits_last[:, PAD_ID] = float('-inf')
        
        probs_raw = torch.softmax(logits_last, dim=-1)
        current_entropy = -(probs_raw * torch.log(probs_raw + 1e-9)).sum(dim=-1).item()
        entropies.append(current_entropy)

        current_temp = TEMPERATURE  # Valore base (es. 0.8 o 1.0)

        if current_entropy < 1.5:
            current_temp = 1.5  # SPINTA CREATIVA: Rompi la sicurezza, rischia di più.
        elif current_entropy > 3.5:
            current_temp = 0.6 # FRENO A MANO: Scegli solo le parole più probabili.

        logits_scaled = logits_last / current_temp

        # Puoi guardare tutta la storia (generated_ids) o solo gli ultimi 20-50 token
        lookback_window = generated_ids[-50:] if len(generated_ids) > 0 else []

        if len(lookback_window) > 0:
            logits_last = apply_repetition_penalty(logits_scaled, lookback_window, penalty=1.2)
        # ---------------------------------------------------------

        probs = torch.softmax(logits_last, dim=-1)

        # --- Sampling ---
        next_id = sample_from_probs(probs, top_k=20, top_p=top_p) # Aggiunto top_k
        
        generated_ids.append(next_id)
        context.append(next_id)

        # Stop condition
        if len(generated_ids) >= MIN_TOKENS_BEFORE_STOP and next_id == END_ID:
            break

    text = sp.DecodeIds(generated_ids)
    text = text.replace("<LINE>", "\n").replace("<END>", "")
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
            top_p=TOP_P,
            block_size=BLOCK_SIZE,
        )
        print(f"\n🪄  Sample {i+1}")
        print(f"   → IDs   : {ids[:20]} … ({len(ids)} token)")
        print(f"   → Text  : {txt}")
        # media dell'entropia (indicatore di “creatività”)
        if ent:
            print(f"   → Entropy (avg): {sum(ent)/len(ent):.3f}")