import torch
import torch.nn.functional as F
from ..FreestyleAI.architecture import WordGramModel
from tokenizer import decode

metadata = torch.load('metadata/bpe-metadata.pt')
merges = metadata["merges"]
max_index = metadata["max_index"]
context_size = metadata["context_size"]
embedding_dim = metadata["embedding_dim"]
vocab_size = metadata["vocab-size"]
reverse_vocab = metadata["reverse_vocab"]

model = WordGramModel(vocab_size)
model.load_state_dict(torch.load('models/bpe-model.pt', map_location="cuda"))
model.to(device = "cuda")
model.eval()


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Totale parametri: {trainable_params:,}")

def compute_entropy(probs):
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy.mean().item()

def generate_sequence_bpe(model, reverse_vocab, max_tokens=50, temperature=1.0, device='cuda'):
    model.eval()
    context = [0]  # definisci questo come il token di inizio (se ne hai uno)
    generated_ids = []
    total_entropy = 0
    steps = 0

    for _ in range(max_tokens):
        x = torch.tensor([context], dtype=torch.long).to(device)
        logits, _ = model(x)
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)

        entropy = compute_entropy(probs)
        total_entropy += entropy
        steps += 1

        # Adatta temperatura dinamica
        if entropy < 2.2:
            temperature += 0.2
        elif entropy > 4.0:
            temperature = max(0.5, temperature - 0.1)

        next_id = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_id)
        context = context[1:] + [next_id] if len(context) >= context_size else context + [next_id]

        # opzionale: fermati a un token di fine
        if reverse_vocab.get(next_id) == "<LINE>":
            break

    return generated_ids

if __name__ == "__main__":
    for _ in range(10):
        output_ids = generate_sequence_bpe(model, reverse_vocab)
        text = decode(output_ids, reverse_vocab)
        print("Testo:", text)