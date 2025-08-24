import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import NanoGPT

def demonstrate_epoch_vs_batch():
    print("=== Understanding Epochs, Batches, and Steps ===")
    print("Key concepts in training loops\n")
    
    total_examples = 1000
    batch_size = 50
    
    batches_per_epoch = total_examples // batch_size
    
    print(f"Dataset size: {total_examples} examples")
    print(f"Batch size: {batch_size} examples per batch")
    print(f"Batches per epoch: {batches_per_epoch}")
    
    print(f"\n1 EPOCH = seeing ALL {total_examples} examples once")
    print(f"1 STEP = processing 1 batch ({batch_size} examples)")
    print(f"So 1 epoch = {batches_per_epoch} steps")
    
    epochs = 3
    total_steps = epochs * batches_per_epoch
    
    print(f"\nTraining for {epochs} epochs:")
    print(f"Total steps: {total_steps}")
    print(f"Total examples processed: {total_steps * batch_size} (with repetition)")
    
    print("\nSimulated training progress:")
    step = 0
    for epoch in range(epochs):
        print(f"\n--- EPOCH {epoch + 1} ---")
        for batch in range(batches_per_epoch):
            step += 1
            if batch % 5 == 0:
                print(f"  Step {step:2d} (Epoch {epoch+1}, Batch {batch+1}): Processing examples {batch*batch_size}-{(batch+1)*batch_size-1}")

def demonstrate_mini_training():
    print("\n=== Mini Training Loop in Action ===")
    print("Watch a model actually learn on fake data\n")
    
    vocab_size = 4
    model = NanoGPT(vocab_size=vocab_size, d_model=8, n_heads=2, n_layers=1, max_seq_len=3)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    fake_data = [
        (torch.tensor([[0, 1]]), torch.tensor([[2]])),
        (torch.tensor([[1, 2]]), torch.tensor([[3]])),
        (torch.tensor([[2, 3]]), torch.tensor([[0]])),
        (torch.tensor([[0, 1]]), torch.tensor([[2]])),
        (torch.tensor([[1, 2]]), torch.tensor([[3]])),
        (torch.tensor([[2, 3]]), torch.tensor([[0]])),
    ]
    
    print(f"Training data (pattern to learn):")
    for i, (inp, target) in enumerate(fake_data):
        print(f"  {inp[0].tolist()} -> {target[0].item()}")
    
    losses = []
    accuracies = []
    
    print(f"\nTraining for 20 epochs...")
    
    for epoch in range(20):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in fake_data:
            model.train()
            logits = model(inputs)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted = logits_flat.argmax(1)
            correct += (predicted == targets_flat).sum().item()
            total += targets_flat.size(0)
        
        avg_loss = epoch_loss / len(fake_data)
        accuracy = correct / total
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        if epoch % 5 == 0 or epoch == 19:
            print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.3f}")
    
    print(f"\nTesting what the model learned:")
    model.eval()
    with torch.no_grad():
        test_cases = [
            torch.tensor([[0, 1]]),
            torch.tensor([[1, 2]]), 
            torch.tensor([[2, 3]])
        ]
        
        for test_input in test_cases:
            logits = model(test_input)
            predicted = logits.argmax(-1)
            probs = torch.softmax(logits[0, -1], dim=0)
            
            print(f"Input {test_input[0].tolist()} -> Predicted: {predicted[0, -1].item()}")
            print(f"  Confidence: {probs.max().item():.3f}")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_validation():
    print("\n=== Understanding Validation ===")
    print("Why we need separate data to test the model\n")
    
    vocab_size = 3
    
    train_data = [
        (torch.tensor([[0]]), torch.tensor([[1]])),
        (torch.tensor([[1]]), torch.tensor([[2]])),
        (torch.tensor([[2]]), torch.tensor([[0]])),
    ] * 10
    
    val_data = [
        (torch.tensor([[0]]), torch.tensor([[1]])),
        (torch.tensor([[1]]), torch.tensor([[2]])),
        (torch.tensor([[2]]), torch.tensor([[0]])),
    ] * 2
    
    print("Training set: Model learns from this")
    print("Validation set: We test on this (model never trained on it)")
    print("If val performance is good, model truly learned the pattern!")
    
    model = nn.Linear(3, 3)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    print(f"\nTraining...")
    
    for epoch in range(15):
        model.train()
        train_loss = 0
        for inputs, targets in train_data:
            inputs_onehot = torch.zeros(1, 3)
            inputs_onehot[0, inputs.item()] = 1
            
            logits = model(inputs_onehot)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_data:
                inputs_onehot = torch.zeros(1, 3)
                inputs_onehot[0, inputs.item()] = 1
                
                logits = model(inputs_onehot)
                loss = criterion(logits, targets)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_data))
        val_losses.append(val_loss / len(val_data))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"\nFinal results:")
    print(f"Training loss: {train_losses[-1]:.4f}")
    print(f"Validation loss: {val_losses[-1]:.4f}")
    
    if abs(train_losses[-1] - val_losses[-1]) < 0.1:
        print("✅ Good! Train and val losses are similar - model generalized well")
    else:
        print("⚠️ Train and val losses differ - might be overfitting")

if __name__ == "__main__":
    demonstrate_epoch_vs_batch()
    demonstrate_mini_training() 
    demonstrate_validation()
    
    print("\n" + "="*60)
    print("TAKEAWAY: Training loop is just repeating these steps:")
    print("1. Feed batch through model (forward)")
    print("2. Calculate how wrong we are (loss)")
    print("3. Calculate gradients (backward)")
    print("4. Update weights (optimizer)")
    print("5. Repeat until model learns the pattern!")