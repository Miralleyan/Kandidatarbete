import torch
from torch import optim

torch.manual_seed(1)

LR = 0.1
mu = 40

p0 = torch.tensor(0.1, requires_grad=True)
p = p0
opt = optim.Adam([p], lr=LR)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=0, verbose=True)
epochs = 1000

for epoch in range(epochs):
    opt.zero_grad()
    loss = p*(1-p) + mu * (max(0, p-1) + max(0, -p)) ** 2
    loss.backward()
    opt.step()
    if epoch % 10 == 9: print(float(loss))

print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tP = {p.item():.4f}')
mu *= 2

