import torch.nn as nn
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu' #use GPU if available
n_head = 4 #n_head = n_embd // head_size

class Head(nn.Module): 
    """ one head of self-attention with FLASH ATTENTION """
    def __init__(self, head_size, emb_size, dropout):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.dropout_val = dropout # Salviamo il valore float, non il layer nn.Dropout

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # --- FLASH ATTENTION MAGIC ---
        # Questa funzione sceglie automaticamente l'implementazione più veloce
        # (FlashAttention v2, MemoryEfficient, o Math) in base alla tua GPU.
        # is_causal=True applica automaticamente la maschera triangolare.
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout_val if self.training else 0.0,
            is_causal=True
        )
        
        return out
    
class MultiHeadAttention(nn.Module): 
    """multiple heads running in parallel"""
    def __init__(self, head_size : int, emb_size : int, dropout : int):
      super().__init__()
      self.heads = nn.ModuleList([
         Head(head_size, emb_size, dropout) for _ in range(n_head)
      ])
      self.proj = nn.Linear(head_size * n_head, emb_size)
      self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
      out = torch.cat([h(x) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out
    
class FeedForward(nn.Module): 

    def __init__(self, n_emb, dropout):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_emb, 4 * n_emb),
         nn.GELU(),
         nn.Linear(4 * n_emb, n_emb),
         nn.Dropout(dropout)
      )    

    def forward(self, x):
      return self.net(x)

class Block(nn.Module): 

    def __init__(self, n_embd : int, dropout : int):
      super().__init__()
      head_size = n_embd // n_head
      self.sa = MultiHeadAttention(head_size, n_embd, dropout)
      self.ffwd = FeedForward(n_embd, dropout) 
      self.ln1 = nn.LayerNorm(n_embd)
      self.ln2 = nn.LayerNorm(n_embd)

    def forward(self , x): 
      x = x + self.sa(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))
      return x

class WordGramModel(nn.Module):
    embedding_size : int
    vocab_size : int 
    context_size : int

    def __init__(self, vocab_size : int, emb_size : int, context_size : int, dropout : int = 0):
        super().__init__()
        self.embedding_size = emb_size
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        self.positional_embedding = nn.Embedding(context_size, emb_size)
        self.blocks = nn.Sequential(
            Block(emb_size , dropout),
            Block(emb_size , dropout),
            Block(emb_size , dropout),
            Block(emb_size , dropout),
            nn.LayerNorm(emb_size)
        )
        self.output = nn.Sequential(
            nn.Dropout(.15),
            nn.Linear(emb_size, vocab_size)
        )

    def forward(self, x, targets=None, return_activations=False):
        activations = []

        B, T = x.shape

        # --- Embedding + Positional ---
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.positional_embedding(torch.arange(T, device=x.device))
        h = tok_emb + pos_emb
        if return_activations:
            activations.append(h.detach())

        # --- Blocks ---
        h = self.blocks(h)
        if return_activations:
            activations.append(h.detach())

        # --- Output layer ---
        logits = self.output(h)
        if return_activations:
            activations.append(logits.detach())

        # --- Loss ---
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        if return_activations:
            return logits, loss, activations
        else:
            return logits, loss
        
