from torch import nn
import torch
from torch.optim.lr_scheduler import*

#model = [nn.parameter.Parameter(torch.randn(2, 2, requires_grad=True))]
po=torch.tensor(0.7,requires_grad=True)
p=po
lr=2
optimizer = torch.optim.SGD([p], lr)
scheduler = MultiplicativeLR(optimizer,lambda x: x*0.7)

optimizer.param_groups[0]["params"][0]
def vart(p):
    return p.item()*(1-p.item())+ 10**8*(p.item()<0 or p.item()>1)

for step in range(20):
    optimizer.zero_grad()
    var=p*(1-p)+ 10**8*(p.item()<0 or p.item()>1)
    var.backward()
    print(vart(p), vart(po))
    print(vart(p)>vart(po))
    if vart(p)>vart(po):
        scheduler.step()
    else:
        po=torch.tensor(p.item(),requires_grad=True)
        optimizer.step()
        
    if step == step:
        print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')