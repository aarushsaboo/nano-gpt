import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        seq_len = Q.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)
        
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_linear(attention)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        
        ff_out = self.feed_forward(self.ln2(x))
        x = x + ff_out
        
        return x

if __name__ == "__main__":
    d_model = 8
    n_heads = 2
    seq_len = 4
    batch_size = 1
    
    transformer_block = TransformerBlock(d_model, n_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x[0]}")
    
    output = transformer_block(x)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output[0]}")
    
    print("Transformer block does:")
    print("1. Layer norm + attention + residual connection")
    print("2. Layer norm + feed-forward + residual connection")
    print("Residual connections: output = input + transformation")
    print("This helps with training stability")