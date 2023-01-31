#2a)

#Optimizar med interior penalty method, med 9 iterationer får vi bra värde, mer än det och den gungar lite fram
#och tillbaka

import torch
from torch import nn

'''
p0=torch.tensor(-10.1,requires_grad=True)
p=p0
#optimizer=torch.optim.SGD([p],lr=0.001)
optimizer=torch.optim.Adam([p],lr=0.1)
steps=200
mu=20

for st in range(9):#Bäst vid 9 men också bra med över 100
    for step in range(steps):
        optimizer.zero_grad() # otherwise, the grads on steps will sum up
        var = p*(1-p) +mu*max(0,p-1)**2+mu*max(0,-p)**2 # the minus so that to minimise var
        var.backward()
        #Save p
        optimizer.step()
        #Compare new p with old, if new is larger decrease steppsize
        #0.7 is magic stepsize value
        if step == steps-1:
            print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')
    mu=mu*2
    print(mu)
'''


import torch
from torch import nn

p0=torch.tensor(-10.1,requires_grad=True)
p=p0
#optimizer=torch.optim.SGD([p],lr=0.001)
lr=0.1
optimizer=torch.optim.Adam([p],lr)
steps=200


for step in range(steps):
    optimizer.zero_grad() # otherwise, the grads on steps will sum up
    var = p*(1-p)# the minus so that to minimise var
    var.backward()
    #Save p
    po=torch.tensor(p.item(),requires_grad=True)
    optimizer.step()
    if po.item()*(1-po.item())<p.item()*(1-p.item()) or p.item()<0 or p.item()>1:
        for g in optimizer.param_groups:
            g['lr'] = lr/2
        print(po)
        p=torch.tensor(po.item(),requires_grad=True)
    
    #Compare new p with old, if new is larger decrease steppsize
    #0.7 is magic stepsize value
    if step == steps-1:
        print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')


