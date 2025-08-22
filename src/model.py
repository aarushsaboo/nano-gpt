import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos)
        
        return token_emb + pos_emb

class MultiHeadSelfAttention(nn.Module):
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
        
        # Create Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attention = self.scaled_dot_product_attention(Q, K, V)
        
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_linear(attention)
    
    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask - prevent looking at future tokens
        seq_len = Q.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

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
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        
        ff_out = self.feed_forward(self.ln2(x))
        x = x + ff_out
        
        return x

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, max_seq_len=256):
        super().__init__()
        self.embedding = PositionalEmbedding(vocab_size, d_model, max_seq_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits

if __name__ == "__main__":
    print("=== Testing Transformer Model ===")
    
    vocab_size = 15  # From our Shakespeare sample
    batch_size = 2
    seq_len = 8
    
    model = NanoGPT(vocab_size=vocab_size, d_model=64, n_heads=2, n_layers=2)
    
    # Create dummy input (like what our dataset produces)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}") # you're calculating the no. of elements for every parameter.. Important because more parameters = more memory and longer training time.
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        logits = model(dummy_input)
    
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]")
    
    print("\n=== Understanding the Output ===")
    print("Logits = raw predictions for each position")
    print("Shape: [batch, sequence, vocabulary]")
    print(f"For each position, we get {vocab_size} scores (one per character)")
    
    # Show what the model predicts for first example
    first_example_logits = logits[0]  # [seq_len, vocab_size]
    
    print(f"\nFirst example predictions:")
    print(f"Position 0 logits: {first_example_logits[0]}")
    print(f"Highest score at position 0: character {first_example_logits[0].argmax().item()}")
    
    print("\nNext step: Train this model on our Shakespeare data!")