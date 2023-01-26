import torch
from torch import optim

torch.manual_seed(1)

LR = 0.1

p = torch.tensor(0.41, requires_grad=True)
opt = optim.Adam([p], lr=LR)

for epoch in range(50):
    opt.zero_grad()
    loss = p*(1-p)
    loss.backward()
    opt.step()

    with torch.no_grad():
        p.clamp_(0, 1)

    print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tP = {p.item():.4f}')
