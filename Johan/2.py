#2a)

#Omptimerar med exterior penatly method, blir bara max istället för min

import torch
from torch import nn
max(0,1)
p0=torch.tensor(0.2,requires_grad=True)
p=p0
optimizer=torch.optim.SGD([p],lr=0.01)
steps=10
mu=2

for st in range(5):
    for step in range(steps):
        optimizer.zero_grad() # otherwise, the grads on steps will sum up
        var = p*(1-p) -mu*torch.log(1-p)-mu*torch.log(p) # the minus so that to minimise var
        var.backward()
        optimizer.step()
        if step == steps-1:
            print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')
    mu=mu/2
    print(mu)

