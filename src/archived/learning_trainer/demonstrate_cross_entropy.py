import torch
import torch.nn as nn
import torch.nn.functional as F

def demonstrate_cross_entropy():
    print("=== Understanding Cross-Entropy Loss ===")
    print("This is how we measure 'how wrong' our model's predictions are\n")
    
    vocab_size = 3
    char_names = ['a', 'b', 'c']
    
    print("Scenario: Model needs to predict next character from 'a', 'b', 'c'")
    print("True answer: 'b' (index 1)\n")
    
    print("Case 1: Model is confident and CORRECT")
    logits_confident_correct = torch.tensor([0.1, 5.0, 0.1])
    probs = F.softmax(logits_confident_correct, dim=0)
    print(f"Raw scores (logits): {logits_confident_correct}")
    print(f"Probabilities: a={probs[0]:.3f}, b={probs[1]:.3f}, c={probs[2]:.3f}")
    
    target = torch.tensor(1)
    loss = F.cross_entropy(logits_confident_correct.unsqueeze(0), target.unsqueeze(0))
    print(f"Loss: {loss.item():.4f} (LOW - good!)\n")
    
    print("Case 2: Model is confident but WRONG")
    logits_confident_wrong = torch.tensor([5.0, 0.1, 0.1])
    probs = F.softmax(logits_confident_wrong, dim=0)
    print(f"Raw scores (logits): {logits_confident_wrong}")
    print(f"Probabilities: a={probs[0]:.3f}, b={probs[1]:.3f}, c={probs[2]:.3f}")
    
    loss = F.cross_entropy(logits_confident_wrong.unsqueeze(0), target.unsqueeze(0))
    print(f"Loss: {loss.item():.4f} (HIGH - bad!)\n")
    
    print("Case 3: Model is UNCERTAIN")
    logits_uncertain = torch.tensor([1.0, 1.0, 1.0])
    probs = F.softmax(logits_uncertain, dim=0)
    print(f"Raw scores (logits): {logits_uncertain}")
    print(f"Probabilities: a={probs[0]:.3f}, b={probs[1]:.3f}, c={probs[2]:.3f}")
    
    loss = F.cross_entropy(logits_uncertain.unsqueeze(0), target.unsqueeze(0))
    print(f"Loss: {loss.item():.4f} (MEDIUM - not great, not terrible)\n")
    
    print("Key Insight: Cross-entropy loss punishes wrong confident predictions the most!")

def demonstrate_batch_loss():
    print("\n=== Batch Loss Calculation ===")
    print("In reality, we process multiple examples at once\n")
    
    batch_size = 2
    seq_length = 4
    vocab_size = 5
    
    logits = torch.randn(batch_size, seq_length, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"Logits shape: {logits.shape} [batch, sequence, vocabulary]")
    print(f"Targets shape: {targets.shape} [batch, sequence]")
    
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    print(f"Flattened logits: {logits_flat.shape}")
    print(f"Flattened targets: {targets_flat.shape}")
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    print(f"Average loss across all predictions: {loss.item():.4f}")
    
    print("\nThis single number tells us how well our model predicts ALL characters!")

if __name__ == "__main__":
    demonstrate_cross_entropy()
    demonstrate_batch_loss()
    
    print("\n" + "="*60)
    print("TAKEAWAY: Loss function converts 'how wrong are we?' into a number")
    print("Training tries to make this number as small as possible!")