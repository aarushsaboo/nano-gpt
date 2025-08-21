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
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2) #.view( batch_size, seq_len, self.n_heads, self.head_dim) just reshapes that last d_model dimension into two parts: (n_heads, head_dim).
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        seq_len = Q.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)
        
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model) # after .transpose(), PyTorch sometimes gives a non contiguous tensor ( means that the numbers in memory aren’t stored in the standard row-major order)
        # that's why you do contiguous first -> it makes a clean copy in memory -> then view() works safely
        return self.out_linear(attention)

if __name__ == "__main__":
    d_model = 8
    n_heads = 2
    seq_len = 4
    batch_size = 1
    
    multihead_attention = MultiHeadAttention(d_model, n_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"Model dimensions: {d_model}")
    print(f"Number of heads: {n_heads}")
    print(f"Each head dimension: {d_model // n_heads}")
    
    output = multihead_attention(x)
    print(f"Output shape: {output.shape}")
    
    print("Multi-head means running multiple attention mechanisms in parallel")
    print("Each head learns different types of relationships")
    print("Head 1 might focus on grammar, Head 2 might focus on meaning")
    print("Final output combines all heads together")