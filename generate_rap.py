import torch
import torch.nn.functional as F
from architecture import WordGramModel
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
import random
import re

metadata = torch.load('metadata/eng-word-gram_metadata.pt')
wotoi = metadata['wotoi']
itow = metadata['itow']
context_size = metadata['context_size']
embedding_dim = metadata['embedding_dim']

vocab_size = len(wotoi)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WordGramModel(vocab_size)
model.load_state_dict(torch.load('models/eng-word-gram_model.pt', map_location=device))
model.to(device)
model.eval()

def compute_entropy(probs):
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy.mean().item()

def generate_sequence(seed_words=None, max_words=50, max_lines=None, temperature=1.0):
    if seed_words is None:
        seed_words = ["<START>"]

    context = [wotoi.get(word, wotoi["<UNK>"]) for word in seed_words]
    generated_lines = []
    total_entropy = 0
    steps = 0

    while True:
        generated = []
        for _ in range(max_words):
            x = torch.tensor([context], dtype=torch.long).to(device)
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            entropy = compute_entropy(probs)
            total_entropy += entropy
            steps += 1

            if entropy < 2.2:
                temperature += 0.2
            elif entropy > 4.0:
                temperature = max(0.5, temperature - 0.1)

            next_indices = torch.multinomial(probs, num_samples=2).squeeze()
            next_word = "<UNK>"
            next_ix = next_indices[0].item()
            for idx in next_indices:
                word = itow.get(idx.item(), "<UNK>")
                if word != "<UNK>":
                    next_word = word
                    next_ix = idx.item()
                    break

            generated.append(next_word)
            context = context[1:] + [next_ix]

            if next_word == "<END>":
                break

        line = " ".join(generated).replace("<START>", "").replace("<END>", "").strip()
        generated_lines.append(line)

        if max_lines is not None:
            if len(generated_lines) >= max_lines:
                break
        else:
            break

    return "\n".join(generated_lines)

def generate_song(seed_words=None):
    song = generate_sequence(seed_words, max_words=50, max_lines=15)
    with open("generated_rap_songs.txt", 'w') as f: 
        f.write("\nNew Song\n" + song + "\n")
    return song

def generate_lyric(seed_words=None):
    return generate_sequence(seed_words, max_words=50)


if __name__ == "__main__":
    for _ in range(10):
        print(generate_lyric())

"""View the space rapresentation of the words to check for generalizations"""
'''top_n = 200
words_sub = []
embedding_sub = []
casual_words = random.sample(list(itow.items()), top_n)

for word in casual_words:
    words_sub.append(word[1])
    embedding_sub.append(embedding_matrix[word[0]])

embedding_sub = torch.stack(embedding_sub).numpy()

#Dimensionality reduction
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(embedding_sub)

#Plotting
plt.figure(figsize=(12, 10))
for idx, (x, y) in enumerate(embedding_2d):
   plt.scatter(x, y, color='blue', alpha=0.6)
   plt.text(x, y, words_sub[idx], fontsize=8)

plt.title("Embedding 2D di parole significative")
plt.grid(True)
plt.show()'''