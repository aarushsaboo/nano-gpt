import torch
import torch.nn as nn

x = torch.randn(2, 5, requires_grad=True)  # input with gradient tracking
y_true = torch.tensor([[1., 0., 0.], [0., 1., 0.]])  # one-hot encoded true targets for 2 samples & 3 classes

# Step 2: define layer
layer = nn.Linear(5, 3) 

# Step 3: forward pass
y_pred = layer(x)  
loss_fn = nn.CrossEntropyLoss() # this is when you configure it! You don't pass your data.. You pass y_pred & y_true when you call it
loss = loss_fn(y_pred, y_true.argmax(dim=1))  

print("Loss value:", loss.item())

# Step 4: backward pass (compute gradients!)
loss.backward()
# check gradients
print(layer.weight)
print("\nGradients for weights:")
print(layer.weight.grad)

print("\nGradients for biases:")
print(layer.bias.grad)
