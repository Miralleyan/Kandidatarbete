import torch
from torch import optim

torch.manual_seed(1)

LR = 0.1

p = torch.tensor(0.41, requires_grad=True)
q = torch.tensor(0.61, requires_grad=True)
opt = optim.Adam([p], lr=LR)
epochs = 50

for epoch in range(epochs):
    opt.zero_grad()
    loss = p*q
    loss.backward()
    opt.step()

    with torch.no_grad():
        p.clamp_(0, 1)
        q.clamp_(0, 1)

    print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tP = {p.item():.4f}\tQ = {q.item():.4f}')
