import torch
import torch.nn as nn

if __name__ == "__main__":
    vocab_size = 5
    embedding_dim = 3
    
    embedding = nn.Embedding(vocab_size, embedding_dim) # embedding layer
    print(f"Embedding: {vocab_size} tokens -> {embedding_dim} dimensions")
    
    token_ids = torch.tensor([0, 1, 2])
    print(f"Input token IDs: {token_ids}")
    
    embedded = embedding(token_ids)
    print(f"Output shape: {embedded.shape}")
    print(f"Embedded vectors:\n{embedded}")
    
    print("Embeddings convert integer tokens to vectors that neural networks can use")
    print("Each token gets its own learnable vector")