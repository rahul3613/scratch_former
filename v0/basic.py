import torch
import torch.nn as nn


class Transf(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super(Transf, self).__init__()
        
        self.embd_dim = 10
        self.embedding = nn.Embedding(vocab_size, self.embd_dim)
        self.pos_embd = nn.Embedding(max_seq_len, self.embd_dim)
        
        self.Q = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.K = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.V = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        
        self.op_proj = nn.Linear(self.embd_dim, vocab_size, bias=False)
        
        
    def forward(self, x):
        
        seq_len = x.shape[-1]
        positions = torch.arange(seq_len)  # -> (seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0) # Lower triangular matrix with -> (seq_len, seq_len) -> (1, seq_len, seq_len)
        
        pos_embd = self.pos_embd(positions).unsqueeze(0)        # (seq_len)  ->  (seq_len, embd_dim)  ->  (1, seq_len, embd_dim)
        x = self.embedding(x) + pos_embd    # (batch, seq_len)  ->  (batch, seq_len, embd_dim)
        
        Q = self.Q(x)                      # (, seq_len, embd_dim)  ->  (, seq_len, embd_dim)
        K = self.K(x)
        V = self.V(x)
        
        score = Q @ K.transpose(-2, -1)     # (, seq_len, embd_dim)  x  (, embd_dim, seq_len)  ->  (, seq_len, seq_len)
        score = score.masked_fill(mask==0, -1e9)
        score /= self.embd_dim ** 0.5
        attn = torch.softmax(score, dim=-1) # (, seq_len, seq_len)  ->  (, seq_len, seq_len)
        out = attn @ V                     # (, seq_len, seq_len)  x  (, seq_len, embd_dim)  ->  (, seq_len, embd_dim)
        
        logits = self.op_proj(out)         # (, seq_len, embd_dim)  ->  (, seq_len, op_vocab)
        
        return logits
    
    
