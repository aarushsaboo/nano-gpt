import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model) # learnable linear projections, input size & output size
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)) # causal mask
        scores = scores.masked_fill(mask == 0, float('-inf')) # block attention to future tokens. mask == 0 marks forbidden spots( upper triangle ). 
        
        weights = F.softmax(scores, dim=-1) # effectively removes those connections, because Softmax turns each row of scores into a probability distribution. So now, for each token, we know how much it attends to each other token.
        output = torch.matmul(weights, V) # Now we mix values (V) according to attention weights.
        
        return output, weights

if __name__ == "__main__":
    d_model = 4 # or embedding_dim. 
    seq_len = 3
    batch_size = 1
    
    attention = SimpleSelfAttention(d_model)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x[0]}")
    
    output, weights = attention(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights:\n{weights[0]}")
    
    print("Each row shows how much each position attends to other positions")
    print("Row 0: position 0 can only look at position 0")
    print("Row 1: position 1 can look at positions 0,1")
    print("Row 2: position 2 can look at positions 0,1,2")