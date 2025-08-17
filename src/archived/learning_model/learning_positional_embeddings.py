import torch
import torch.nn as nn

if __name__ == "__main__":
    vocab_size = 5
    embedding_dim = 3
    max_seq_len = 10
    
    token_embedding = nn.Embedding(vocab_size, embedding_dim) # is a lookup table, we have vocab_size possible tokens & each tokenID maps to a vector of length embedding_dim
    position_embedding = nn.Embedding(max_seq_len, embedding_dim)
    
    tokens = torch.tensor([[0, 1, 2]]) # we always keep a batch dimension, even if it's 1
    seq_len = tokens.size(1) # get the size along the dimension 1( dimension 0 is the batch size = 1, dimension 1 is the sequene length = 3)

    print(f"Tokens: {tokens}")
    
    token_emb = token_embedding(tokens) # maps each integer ( token ID ) to a vector of length embedding_dim
    print("Here's your token_embeddings", token_emb, "\n")
    print(f"Token embeddings shape: {token_emb.shape}") # output shape = [1, 3, 3] = [batch, seq_len, embed_dim] 
    
    positions = torch.arange(0, seq_len).unsqueeze(0)
    print(f"Position indices: {positions}")
    
    pos_emb = position_embedding(positions)
    print(f"Position embeddings shape: {pos_emb.shape}")
    
    final_embedding = token_emb + pos_emb
    print(f"Final embedding shape: {final_embedding.shape}")
    
    print("Position embedding tells model WHERE each token appears")
    print("Same token at different positions gets different final embeddings")