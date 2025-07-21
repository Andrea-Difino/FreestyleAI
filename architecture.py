import torch.nn as nn
import torch

class WordGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(emb.size(0), -1)

        h1 = torch.tanh(self.ln1(self.fc1(emb)))
        h1 = self.dropout(h1)

        h2 = torch.tanh(self.ln2(self.fc2(h1)))
        h2 = self.dropout(h2)

        logits = self.output(h2)
        return logits