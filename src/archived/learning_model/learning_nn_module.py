import torch
import torch.nn as nn

class SimpleLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    layer = SimpleLayer(input_size=5, output_size=3)
    x = torch.randn(2, 5)
    print(f"Input shape: {x.shape}")
    
    output = layer(x)
    print(f"Output shape: {output.shape}")
    
    print("nn.Module lets PyTorch track this as a neural network component")
    print("super().__init__() registers it properly for training")