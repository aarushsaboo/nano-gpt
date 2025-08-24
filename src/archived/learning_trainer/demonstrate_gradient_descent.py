import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def demonstrate_gradient_descent():
    print("=== Understanding How Optimizers Work ===")
    print("The optimizer adjusts model weights to reduce loss\n")
    
    x = torch.tensor(0.0, requires_grad=True)
    target = 3.0
    
    optimizer = optim.SGD([x], lr=0.1)
    
    losses = []
    x_values = []
    
    print("Starting optimization...")
    print(f"Initial guess: x = {x.item():.3f}")
    print(f"Target: x = {target}")
    
    for step in range(20):
        loss = (x - target) ** 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        x_values.append(x.item())
        
        if step % 5 == 0:
            print(f"Step {step:2d}: x = {x.item():.3f}, loss = {loss.item():.3f}, gradient = {x.grad.item():.3f}")
    
    print(f"\nFinal result: x = {x.item():.3f} (should be close to 3.0)")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss Over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2) 
    plt.plot(x_values)
    plt.axhline(y=target, color='r', linestyle='--', label='Target')
    plt.title('Parameter Value Over Time')
    plt.xlabel('Step')
    plt.ylabel('x value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_optimizers():
    print("\n=== Comparing Different Optimizers ===")
    print("SGD vs Adam vs AdamW - which learns faster?\n")
    
    target = 5.0
    steps = 50
    
    optimizers_data = {}
    
    for name, opt_class, lr in [('SGD', optim.SGD, 0.1), 
                                ('Adam', optim.Adam, 0.1), 
                                ('AdamW', optim.AdamW, 0.1)]:
        
        x = torch.tensor(0.0, requires_grad=True)
        optimizer = opt_class([x], lr=lr)
        
        losses = []
        x_values = []
        
        for step in range(steps):
            loss = (x - target) ** 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            x_values.append(x.item())
        
        optimizers_data[name] = {'losses': losses, 'x_values': x_values, 'final_x': x.item()}
        print(f"{name}: Final x = {x.item():.3f}, Final loss = {losses[-1]:.6f}")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for name, data in optimizers_data.items():
        plt.plot(data['losses'], label=name)
    plt.title('Loss Comparison')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, data in optimizers_data.items():
        plt.plot(data['x_values'], label=name)
    plt.axhline(y=target, color='r', linestyle='--', label='Target')
    plt.title('Parameter Convergence')
    plt.xlabel('Step')
    plt.ylabel('x value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return optimizers_data

def demonstrate_learning_rates():
    print("\n=== Effect of Learning Rate ===")
    print("Learning rate controls how big steps the optimizer takes\n")
    
    target = 3.0
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        x = torch.tensor(0.0, requires_grad=True)
        optimizer = optim.SGD([x], lr=lr)
        
        losses = []
        x_values = []
        
        for step in range(30):
            loss = (x - target) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            x_values.append(x.item())
        
        plt.subplot(2, 2, i+1)
        plt.plot(x_values, label=f'LR = {lr}')
        plt.axhline(y=target, color='r', linestyle='--', alpha=0.7)
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('Step')
        plt.ylabel('x value')
        plt.grid(True)
        plt.legend()
        
        print(f"LR {lr}: Final x = {x_values[-1]:.3f}")
    
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("- Too small LR: Slow convergence")
    print("- Good LR: Steady progress to target") 
    print("- Too large LR: Might overshoot or oscillate")

if __name__ == "__main__":
    demonstrate_gradient_descent()
    optimizers_data = compare_optimizers()
    demonstrate_learning_rates()
    
    print("\n" + "="*60)
    print("TAKEAWAY: Optimizer is like GPS navigation for your model")
    print("It tells the model which direction to adjust its weights")
    print("AdamW is popular because it adapts to the problem automatically!")