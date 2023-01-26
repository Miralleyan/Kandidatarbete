import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)


# Uppgift 2
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

p = torch.tensor(0.2, requires_grad=True)
q = torch.tensor(0.2, requires_grad=True)
u = torch.tensor(0.2, requires_grad=True)

optimizer = torch.optim.Adam([p, q, u], lr=0.1)

epochs = 100

for epoch in range(epochs):
    loss = (q + u) ** 2 + (p + u) ** 2 + (p + q - 1) ** 2
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tP = {p.item():.4f}\tQ = {q.item():.4f}\tU = {u.item():.4f}')