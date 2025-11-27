# Math mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# We use a tiny vocab: 0-9, +, =, and space (pad)
chars = "0123456789+= "
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

class Config:
    vocab_size = len(chars)
    n_embed = 256
    n_heads = 4          
    block_size = 16    
    dropout = 0.05
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

Config = Config()
print("Using device:", Config.device)

# Standard transformer components
class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.n_embed, head_size, bias=False)
        self.query = nn.Linear(Config.n_embed, head_size, bias=False)
        self.value = nn.Linear(Config.n_embed, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(Config.block_size, Config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        v = self.value(x) # (B,T,C)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)

        # Apply the causal mask
        # 'to ensure that the model only attends to past and present tokens, never future ones.'
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax normalization
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self):
        super().__init__()
        head_size = Config.n_embed // Config.n_heads
        self.sa = MultiHeadAttention(Config.n_heads, head_size)
        self.ffwd = FeedForward(Config.n_embed)
        self.ln1 = nn.LayerNorm(Config.n_embed)
        self.ln2 = nn.LayerNorm(Config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(Config.n_embed, Config.n_embed)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        
        return out
        
        
class RecurrentGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(Config.vocab_size, Config.n_embed)
        self.position_embedding_table = nn.Embedding(Config.block_size, Config.n_embed)
        
        # Shared blocks
        self.shared_sa = MultiHeadAttention(Config.n_heads, Config.n_embed // Config.n_heads)
        self.shared_ffwd = FeedForward(Config.n_embed)
        self.ln1 = nn.LayerNorm(Config.n_embed)
        self.ln2 = nn.LayerNorm(Config.n_embed)
        
        self.ln_f = nn.LayerNorm(Config.n_embed) # final layer norm
        self.lm_head = nn.Linear(Config.n_embed, Config.vocab_size)
        
    def forward(self, idx, targets=None, recur_depth=1):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # 1.Embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=Config.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        # 2. Recurrent Depth loop
        # so instead of : for layer in self.laayers:
        # we do
        for _ in range(recur_depth):
            x = x + self.shared_sa(self.ln1(x))
            x = x + self.shared_ffwd(self.ln2(x))
        
        # 3. Final layer norm and head
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    

