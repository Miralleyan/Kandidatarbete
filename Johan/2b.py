import torch
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import Bounds
'''
p0 = torch.tensor(0.4, requires_grad=True)
p = p0
opt = torch.optim.Adam([p], lr=0.1)
steps = 100

for step in range(steps):
    opt.zero_grad()
    var=p*(1-p)
    var.backward()
    opt.step()
    with torch.no_grad():
        p.clamp_(0,1)
    if step % 10 == 9:
        print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')

'''

def var(p):
    return p*(1-p)
res = minimize_scalar(var,bounds=(0,1), method="bounded")
print(res.x,res.fun)
print(res)