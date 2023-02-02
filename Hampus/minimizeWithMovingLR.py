import torch
from torch import optim

torch.manual_seed(1)

LR = 10
mu = 10**6

p0 = torch.tensor(0.3, requires_grad=True)
p = p0
opt = optim.Adam([p], lr=LR)
epochs = 100

for epoch in range(epochs):
    while True:
        opt.zero_grad()
        loss = p * (1 - p)
        loss.backward()
        pOld = p.clone()
        opt.step()
        if 1 >= p >= 0:
            break
        p = pOld.clone()
        for g in opt.param_groups:
            g['lr'] *= 0.7
            #print(g['lr'], epoch, float(p))
    if True: print(float(p))

print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tP = {p.item():.4f}')
mu *= 2

