import torch

p = torch.tensor(0.2, requires_grad=True)
q = torch.tensor(0.2, requires_grad=True)
u = torch.tensor(0.2, requires_grad=True)

opt = torch.optim.Adam([p, q, u], lr=0.1)

for epoch in range(500):
    opt.zero_grad()
    loss = (q+u)**2 + (p+u)**2 + (p+q -1)**2
    loss.backward()
    opt.step()
    print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tP = {p.item():.4f}\tQ = {q.item():.4f}\tU = {u.item():.4f}')