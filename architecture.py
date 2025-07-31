import torch.nn as nn
import torch
import torch.nn.functional as F

batch_size = 64 #sequences to be processed in parallel
block_size = 32 #number of words to be processed in parallel
device = 'cuda' if torch.cuda.is_available() else 'cpu' #use GPU if available
n_embd = 256
n_head = 4 #n_head = n_embd // batch_size
dropout = 0.2

class Head(nn.Module): 
    """one head of self-attention"""
    def __init__(self, head_size):
      super().__init__()
      self.query = nn.Linear(n_embd, head_size, bias=False)
      self.key = nn.Linear(n_embd, head_size, bias=False)
      self.value = nn.Linear(n_embd, head_size, bias=False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

      self.dropout = nn.Dropout(dropout)

    def forward(self, x):
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)
      v = self.value(x)

      #compute attention scores "affinities"
      wei = q @ k.transpose(-2 , -1) * C**-0.5
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      wei = F.softmax(wei,dim=-1) #softmax over rows

      wei = self.dropout(wei)

      out = wei @ v
      return out
    
class MultiHeadAttention(nn.Module): 
    """multiple heads running in parallel"""
    def __init__(self, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
      self.proj = nn.Linear(n_embd, n_embd)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out
    
class FeedForward(nn.Module): 

    def __init__(self, n_emb):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_emb, 4 * n_emb),
         nn.ReLU(),
         nn.Linear(4 * n_emb, n_emb),
         nn.Dropout(.2)
      )    

    def forward(self, x):
      return self.net(x)

class Block(nn.Module): 

    def __init__(self, n_embd):
      super().__init__()
      head_size = n_embd // n_head
      self.sa = MultiHeadAttention(head_size)
      self.ffwd = FeedForward(n_embd) 
      self.ln1 = nn.LayerNorm(n_embd)
      self.ln2 = nn.LayerNorm(n_embd)

    def forward(self , x): 
      x = self.ln1(x + self.sa(x))
      x = self.ln2(x + self.ffwd(x))
      return x

class WordGramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd),
            Block(n_embd),
            Block(n_embd),
            nn.LayerNorm(n_embd)
        )
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_embd, vocab_size)
        )

    def forward(self, x, targets = None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.positional_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        logits = self.output(x)
        
        if targets == None: 
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss