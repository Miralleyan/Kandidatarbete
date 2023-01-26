#2a)

#Optimizar med interior penalty method, med 9 iterationer får vi bra värde, mer än det och den gungar lite fram
#och tillbaka

import torch
from torch import nn
max(0,1)
p0=torch.tensor(-10.1,requires_grad=True)
p=p0
#optimizer=torch.optim.SGD([p],lr=0.001)
optimizer=torch.optim.Adam([p],lr=0.1)
steps=200
mu=20

for st in range(9):
    for step in range(steps):
        optimizer.zero_grad() # otherwise, the grads on steps will sum up
        var = p*(1-p) +mu*max(0,p-1)**2+mu*max(0,-p)**2 # the minus so that to minimise var
        var.backward()
        optimizer.step()
        if step == steps-1:
            print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')
    mu=mu*2
    print(mu)