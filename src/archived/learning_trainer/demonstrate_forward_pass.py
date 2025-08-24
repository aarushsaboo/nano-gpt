import torch
import torch.nn as nn
from model import NanoGPT
from tokenizer import CharacterTokenizer

def demonstrate_forward_pass():
    print("=== Understanding Forward Pass ===")
    print("Forward pass = feeding data through the model to get predictions\n")
    
    vocab_size = 10
    model = NanoGPT(vocab_size=vocab_size, d_model=16, n_heads=2, n_layers=1, max_seq_len=8)
    
    batch_size = 1
    seq_length = 4
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"Input shape: {input_tokens.shape}")
    print(f"Input tokens: {input_tokens[0].tolist()}")
    
    print("\nForward pass steps:")
    
    embedded = model.embedding(input_tokens)
    print(f"1. After embedding: {embedded.shape} [batch, seq, d_model]")
    
    x = embedded
    for i, block in enumerate(model.transformer_blocks):
        x = block(x)
        print(f"2.{i+1} After transformer block {i+1}: {x.shape}")
    
    x = model.ln_final(x)
    print(f"3. After final layer norm: {x.shape}")
    
    logits = model.output_projection(x)
    print(f"4. Final logits: {logits.shape} [batch, seq, vocab]")
    
    print(f"\nLogits for position 0: {logits[0, 0, :].tolist()}")
    print("These are raw scores for each possible next character")
    
    probs = torch.softmax(logits[0, 0, :], dim=0)
    print(f"Probabilities: {probs.tolist()}")
    print("Highest probability character:", probs.argmax().item())

def demonstrate_backward_pass():
    print("\n=== Understanding Backward Pass ===")
    print("Backward pass = calculating gradients to update weights\n")
    
    model = nn.Linear(2, 1)
    
    print("Model weights before training:")
    print(f"Weight: {model.weight.data}")
    print(f"Bias: {model.bias.data}")
    
    x = torch.tensor([[1.0, 2.0]])
    y_true = torch.tensor([[5.0]])
    
    print(f"\nInput: {x}")
    y_pred = model(x)
    print(f"Model prediction: {y_pred.item():.3f}")
    print(f"True target: {y_true.item():.3f}")
    
    loss = nn.MSELoss()(y_pred, y_true)
    print(f"Loss: {loss.item():.3f}")
    
    print("\nBackward pass:")
    loss.backward()
    
    print(f"Weight gradient: {model.weight.grad}")
    print(f"Bias gradient: {model.bias.grad}")
    
    lr = 0.1
    with torch.no_grad():
        model.weight -= lr * model.weight.grad
        model.bias -= lr * model.bias.grad
    
    print(f"\nWeights after update (lr={lr}):")
    print(f"New weight: {model.weight.data}")
    print(f"New bias: {model.bias.data}")
    
    model.zero_grad()
    new_pred = model(x)
    new_loss = nn.MSELoss()(new_pred, y_true)
    print(f"\nNew prediction: {new_pred.item():.3f}")
    print(f"New loss: {new_loss.item():.3f}")
    print("Loss should be lower if we moved in the right direction!")

def demonstrate_full_training_step():
    print("\n=== Complete Training Step ===")
    print("Putting it all together: forward + backward + update\n")
    
    vocab_size = 5
    model = NanoGPT(vocab_size=vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randint(0, vocab_size, (2, 3))
    targets = torch.randint(0, vocab_size, (2, 3))
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    
    model.train()
    
    print("\nStep 1: Forward pass")
    logits = model(inputs)
    print(f"Model output shape: {logits.shape}")
    
    print("\nStep 2: Calculate loss")
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    loss = criterion(logits_flat, targets_flat)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nStep 3: Clear old gradients")
    optimizer.zero_grad()
    
    print("\nStep 4: Backward pass (calculate gradients)")
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None and 'embedding' in name:
            print(f"Gradient for {name}: mean = {param.grad.mean().item():.6f}")
        if len([p for n, p in model.named_parameters() if 'embedding' in n]) > 2:
            break
    
    print("\nStep 5: Update weights")
    optimizer.step()
    
    print("\nStep 6: Test new prediction")
    with torch.no_grad():
        new_logits = model(inputs)
        new_logits_flat = new_logits.view(-1, new_logits.size(-1))
        new_loss = criterion(new_logits_flat, targets_flat)
        print(f"New loss: {new_loss.item():.4f}")
        
        if new_loss.item() < loss.item():
            print("✅ Loss decreased - model learned something!")
        else:
            print("⚠️  Loss increased - might need different learning rate")

if __name__ == "__main__":
    demonstrate_forward_pass()
    demonstrate_backward_pass()
    demonstrate_full_training_step()
    
    print("\n" + "="*60)
    print("TAKEAWAY:")
    print("Forward pass: Input → Model → Predictions")
    print("Backward pass: Loss → Gradients → Weight updates")
    print("This cycle repeats thousands of times during training!")