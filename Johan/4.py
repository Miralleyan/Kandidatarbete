import torch
from torch import nn

p = torch.tensor(0.2, requires_grad=True) 
q = torch.tensor(0.2, requires_grad=True) 
u1 = torch.tensor(0.2, requires_grad=True) 
u2 = torch.tensor(0.2, requires_grad=True) 
u3 = torch.tensor(0.2, requires_grad=True) 
u4 = torch.tensor(0.2, requires_grad=True) 
u5 = torch.tensor(0.2, requires_grad=True) 

opt= torch.optim.Adam([p,q,u1,u2,u3,u4,u5],lr=0.2)
steps=200

for step in range(steps):
    opt.zero_grad()
    loss=p*q+u1*(p+q-1)+u2*(-p)+u3*(p-1)+u4*(-q)+u5*(q-1)
    loss.backward()
    opt.step()
    if step % 10 == 9:
        print(f'Step {step+1: 2}: p={p.item(): 0.4f}, q={q.item(): 0.4f}, u1={u1.item(): 0.4f}, u2={u2.item(): 0.4f}, u3={u3.item(): 0.4f}, u4={u4.item(): 0.4f}, u5={u5.item(): 0.4f} and variance is {-loss.item(): 1.4f}')