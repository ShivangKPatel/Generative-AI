import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch.nn as nn

# ==================== TENSORS ====================
print("=== TENSORS ===")

# Create tensors from lists
tensor_1d = torch.tensor([1, 2, 3, 4])
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(f"1D Tensor: {tensor_1d}")
print(f"2D Tensor:\n{tensor_2d}")

# Create tensors with specific values
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random_tensor = torch.randn(2, 3)
print(f"Zeros:\n{zeros}")
print(f"Random:\n{random_tensor}")

# Tensor operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = a + b
d = torch.matmul(a, b)
print(f"Addition: {c}")
print(f"Dot product: {d}")

# ==================== AUTOGRAD ====================
print("\n=== AUTOGRAD ===")

# Enable gradient computation
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([1.0, 4.0], requires_grad=True)
    
# Perform operations
z = x * y
loss = z.sum()

# Backward pass (compute gradients)
loss.backward()
print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")

# ==================== DATALOADERS ====================
print("\n=== DATALOADERS ===")

# Create dummy dataset
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Create TensorDataset
dataset = TensorDataset(X, y)

# Create DataLoader
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate through batches
for batch_idx, (features, labels) in enumerate(dataloader):
    if batch_idx == 0:
        print(f"Batch features shape: {features.shape}")
        print(f"Batch labels shape: {labels.shape}")
    if batch_idx == 2:
        break

print(f"Total batches: {len(dataloader)}")