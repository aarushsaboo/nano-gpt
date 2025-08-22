import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    d_model = 8
    d_ff = 32
    seq_len = 4
    batch_size = 1
    
    ff = FeedForward(d_model, d_ff)
    
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    print(f"Input dimension: {d_model}")
    print(f"Hidden dimension: {d_ff}")
    
    output = ff(x)
    print(f"Output shape: {output.shape}")
    
    print("Feed-forward expands dimensions then contracts back")
    print("Step 1: 8 -> 32 dimensions (expand)")
    print("Step 2: Apply ReLU activation (add non-linearity)")
    print("Step 3: 32 -> 8 dimensions (contract back)")
    print("This processes each position independently")