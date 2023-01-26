import torch
from torch import optim

torch.manual_seed(1)

LR = 0.1
mu = 20

p0 = torch.tensor(10.1, requires_grad=True)
p = p0
opt = optim.Adam([p], lr=LR)
epochs = 200

for st in range(9):
    for epoch in range(epochs):
        opt.zero_grad()
        loss = p*(1-p) + mu * (max(0, p-1) + max(0, -p)) ** 2
        loss.backward()
        opt.step()

    print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tP = {p.item():.4f}')
    mu *= 2
