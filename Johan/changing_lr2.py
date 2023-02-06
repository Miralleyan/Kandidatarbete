from torch import nn
import torch
from torch.optim.lr_scheduler import*

#model = [nn.parameter.Parameter(torch.randn(2, 2, requires_grad=True))]
po=torch.tensor(0.7,requires_grad=True)
p=po
lr=0.1
optimizer = torch.optim.SGD([p], lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

def vart(p):
    return p.item()*(1-p.item())+ 10**8*(p.item()<0 or p.item()>1)

for epoch in range(20):
    po=torch.tensor(p.item(),requires_grad=True) #save p
    optimizer.zero_grad()
    var=p.item()*(1-p.item())-p*(p.item()<0)+p*(p.item()>0)
    var.backward()
    optimizer.step()
    if vart(p)>vart(po):
        p=torch.tensor(po.item(),requires_grad=True)
        scheduler.step()