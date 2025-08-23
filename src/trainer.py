import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import create_dataloaders
from model import NanoGPT

class Trainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, lr=3e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, inputs, targets):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(inputs)
        
        # Reshape for loss calculation
        # logits: [batch, seq_len, vocab] -> [batch*seq_len, vocab]
        # targets: [batch, seq_len] -> [batch*seq_len]
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        loss = self.criterion(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                logits = self.model(inputs)
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def train(self, epochs=100, print_every=10):
        print(f"Starting training for {epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for inputs, targets in self.train_loader:
                loss = self.train_step(inputs, targets)
                epoch_loss += loss
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            val_loss = self.validate()
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Generate sample text
                sample = self.generate_sample("To be", max_length=50)
                print(f"Sample: '{sample}'")
                print("-" * 60)
    
    def generate_sample(self, start_text="To be", max_length=100, temperature=1.0):
        self.model.eval()
        
        # Encode starting text
        tokens = self.tokenizer.encode(start_text)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Convert to tensor
                input_tensor = torch.tensor([tokens], dtype=torch.long)
                
                # Get predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                tokens.append(next_token)
                
                # Stop if we hit a natural break
                if len(tokens) > len(start_text) and self.tokenizer.decode([next_token]) in '.!?':
                    break
        
        return self.tokenizer.decode(tokens)
    
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    print("=== Training Shakespeare Model ===")
    
    # Create sample data (you can replace with real Shakespeare later)
    sample_text = """To be or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come"""
    
    # Save sample data
    import os
    os.makedirs('../data', exist_ok=True)
    with open('../data/sample.txt', 'w') as f:
        f.write(sample_text)
    
    # Create data loaders
    train_loader, val_loader, tokenizer = create_dataloaders(
        '../data/sample.txt', 
        batch_size=4, 
        seq_length=32
    )
    
    # Create model
    model = NanoGPT(
        vocab_size=len(tokenizer),
        d_model=64,
        n_heads=4,
        n_layers=3,
        max_seq_len=64
    )
    
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Training examples: {len(train_loader.dataset)}")
    print(f"Validation examples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, tokenizer)
    
    # Train the model
    trainer.train(epochs=50, print_every=10)
    
    print("\n=== Final Generation Test ===")
    for prompt in ["To be", "The heart", "To die"]:
        generated = trainer.generate_sample(prompt, max_length=30)
        print(f"'{prompt}' -> '{generated}'")
    
    # Plot training curves
    trainer.plot_losses()