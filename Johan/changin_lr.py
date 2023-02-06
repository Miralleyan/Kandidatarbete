
#Omptmizes the variance of Bernoulli but decreases lr if new values is worse than old.
import torch
from torch import nn


p0=torch.tensor(0.7,requires_grad=True)
p=p0
#optimizer=torch.optim.SGD([p],lr=0.001)
lr=0.2
lrn=lr
optimizer=torch.optim.Adam([p],lr)
steps=10

def vart(p):
    return p.item()*(1-p.item())+ 10**8*(p.item()<0 or p.item()>1)
    #return p.item()*(1-p.item())-p*(p.item()<0)+p*(p.item()>1)


for step in range(steps):
    optimizer=torch.optim.Adam([p],lr)
    optimizer.zero_grad() # otherwise, the grads on steps will sum up
    #var = p*(1-p)+10**8*(p<0 or p>1)
    var=p.item()*(1-p.item())-p*(p.item()<0)+p*(p.item()>0)
    var.backward()
    while True:
        po=torch.tensor(p.item(),requires_grad=True) #save p
        print(p)
        optimizer.step()
        print(p)
        if vart(po)<vart(p):
            print("hej")
            lrn=lrn/2
            #optimizer=torch.optim.Adam([p],lrn)
            for g in optimizer.param_groups:
                g['lr'] = g["lr"]/2
            p=torch.tensor(po.item(),requires_grad=True)
        else:
            
            #for g in optimizer.param_groups:
            #    g['lr'] = lr
            #optimizer=torch.optim.Adam([p],lr)
            break
    #Compare new p with old, if new is larger decrease steppsize
    #0.7 is magic stepsize value
    if step == step:
        print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')
