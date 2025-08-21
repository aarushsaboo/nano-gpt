import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharacterTokenizer

class ShakespeareDataset(Dataset):
    def __init__(self, text_file, tokenizer, seq_length=256, split='train'):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokenizer.build_vocabulary(text)
        self.tokens = self.tokenizer.encode(text)
        
        split_idx = int(0.9 * len(self.tokens))
        if split == 'train':
            self.tokens = self.tokens[:split_idx]
        else:
            self.tokens = self.tokens[split_idx:]
    
    #  when passing the ShakespeareDataset object to a DataLoader function of pytorch, it expects the dataset to follow the Dataset protocol. We need:
    # __len__(self) → returns number of examples in the dataset
    # getitem__(self, idx) → returns the i-th example (input, target)
    
    def __len__(self):
        return len(self.tokens) - self.seq_length
    
    def __getitem__(self, idx):
        input_seq = torch.tensor(self.tokens[idx:idx + self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.tokens[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return input_seq, target_seq

def create_dataloaders(text_file, batch_size=8, seq_length=256):
    tokenizer = CharacterTokenizer()
    
    train_dataset = ShakespeareDataset(text_file, tokenizer, seq_length, 'train')
    val_dataset = ShakespeareDataset(text_file, tokenizer, seq_length, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # we shuffle so the model doesn't learn based on the order of data eg: "To be or not"
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # no need to shuffle because no learning happens in validation
    
    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    print("=== Testing Dataset Component ===")
    
    # Create a tiny example to understand the concept
    sample_text = """To be or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles"""
    print(f"Original text: '{sample_text}'")
    print(f"Text length: {len(sample_text)} characters")
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs('../data', exist_ok=True)
    
    # Save sample text (in real project, you'd use shakespeare.txt)
    with open('../data/sample.txt', 'w') as f:
        f.write(sample_text)
    
    train_loader, val_loader, tokenizer = create_dataloaders('../data/sample.txt', batch_size=2, seq_length=8)
    
    print(f"\nTokenizer vocabulary size: {len(tokenizer)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Show what tokens look like
    all_tokens = tokenizer.encode(sample_text)
    print(f"\nFull text as tokens: {all_tokens}")
    print(f"Total tokens: {len(all_tokens)}")
    
    # Show train/val split
    split_point = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_point]
    val_tokens = all_tokens[split_point:]
    print(f"\nTrain tokens (first 90%): {train_tokens}")
    print(f"Val tokens (last 10%): {val_tokens}")
    
    print("\n=== Understanding Training Examples ===")
    print("Each example: input sequence → target sequence (shifted by 1)")
    
    # Manually show how examples are created
    seq_len = 8 
    for i in range(min(1, len(train_tokens) - seq_len)): # we want to show at most 1 examples, but if the dataset is too small we don't try & show more than are present.
        input_tokens = train_tokens[i:i + seq_len]
        target_tokens = train_tokens[i + 1:i + seq_len + 1]
        
        input_text = tokenizer.decode(input_tokens)
        target_text = tokenizer.decode(target_tokens)
        
        print(f"\nExample {i+1}:")
        print(f"  Input tokens:  {input_tokens}")
        print(f"  Target tokens: {target_tokens}")
        print(f"  Input text:    '{input_text}'")
        print(f"  Target text:   '{target_text}'")
        
        print("  Character-by-character predictions:")
        for j in range(len(input_tokens)):
            input_char = tokenizer.decode([input_tokens[j]])
            target_char = tokenizer.decode([target_tokens[j]])
            print(f"    '{input_char}' → '{target_char}'")
    
    # Show what the DataLoader Actually Returns
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1} shape:")
        print(f"  inputs tensor shape: {inputs.shape}")  # [batch_size, seq_length]
        print(f"  targets tensor shape: {targets.shape}")
        
        print(f"\nBatch {batch_idx + 1} content:")
        for i in range(inputs.size(0)):  # For each example in batch
            input_text = tokenizer.decode(inputs[i].tolist())
            target_text = tokenizer.decode(targets[i].tolist())
            print(f"  Example {i+1} in batch:")
            print(f"    Input:  '{input_text}'")
            print(f"    Target: '{target_text}'")
        
        break  # Just show first batch