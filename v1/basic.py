import torch
import torch.nn as nn


class Transf(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super(Transf, self).__init__()
        
        self.embd_dim = 32
        self.embedding = nn.Embedding(vocab_size, self.embd_dim)
        self.pos_embd = nn.Embedding(max_seq_len, self.embd_dim)
        
        self.Q = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.K = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.V = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.ln1 = nn.LayerNorm(self.embd_dim)
        self.ln2 = nn.LayerNorm(self.embd_dim)
        
        self.ff1 = nn.Linear(self.embd_dim, 4 * self.embd_dim)
        self.ff2 = nn.Linear(4 * self.embd_dim, self.embd_dim)
        
        self.op_proj = nn.Linear(self.embd_dim, vocab_size, bias=False)
        
        
    def forward(self, x):
        
        seq_len = x.shape[-1]
        positions = torch.arange(seq_len, device=x.device)  # -> (seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0) # Lower triangular matrix with -> (seq_len, seq_len) -> (1, seq_len, seq_len)
        
        pos_embd = self.pos_embd(positions).unsqueeze(0)        # (seq_len)  ->  (seq_len, embd_dim)  ->  (1, seq_len, embd_dim)
        x = self.embedding(x) + pos_embd        # (batch, seq_len)  ->  (batch, seq_len, embd_dim)
        
        x_norm = self.ln1(x)
        
        # Attention
        Q = self.Q(x_norm)                      # (, seq_len, embd_dim)  ->  (, seq_len, embd_dim)
        K = self.K(x_norm)
        V = self.V(x_norm)
        
        score = Q @ K.transpose(-2, -1)         # (, seq_len, embd_dim)  x  (, embd_dim, seq_len)  ->  (, seq_len, seq_len)
        score = score.masked_fill(mask==0, -1e9)
        score /= self.embd_dim ** 0.5
        
        attn = torch.softmax(score, dim=-1)     # (, seq_len, seq_len)  ->  (, seq_len, seq_len)
        attn_out = attn @ V                     # (, seq_len, seq_len)  x  (, seq_len, embd_dim)  ->  (, seq_len, embd_dim)
        
        # Feed Forward
        x = attn_out + x                        # Resedual connection
        x_norm2 = self.ln2(x)
        
        out = torch.relu(self.ff1(x_norm2))     # (, seq_len, embd_dim)  ->  (, seq_len, 4*embd_dim)
        out = self.dropout(out)
        out = self.ff2(out)                     # (, seq_len, 4*embd_dim)  ->  (, seq_len, embd_dim)
        out = out + x
        
        logits = self.op_proj(out)              # (, seq_len, embd_dim)  ->  (, seq_len, op_vocab)
        
        return logits
    
    
