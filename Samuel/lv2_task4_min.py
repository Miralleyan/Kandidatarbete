import torch
import numpy as np
from sympy import *
from matplotlib import pyplot as plt

# Introduce slack variables, so that all constraints become equalities
# Slack variables are basically non-negative variables that get added to the 
# inequality constraints, so that they become equalities
pars = torch.nn.Parameter(torch.randn(11))

opt = torch.optim.SGD(params=[pars], lr=0.1)

for epoch in range(1000):
    opt.zero_grad()
    loss = (pars[1]+pars[2]-pars[3]+pars[4])**2 + (pars[0]+pars[2]-pars[5]+pars[6])**2\
        + (pars[0]+pars[1]-1)**2 + (-pars[0]+pars[7])**2 + (pars[0]-1+pars[8])**2\
        + (-pars[1]+pars[9])**2 + (pars[1]-1+pars[10])**2 + pars[2]**2 + pars[3]**2\
        + pars[4]**2 + pars[5]**2
    loss.backward()
    print(pars.grad)
    opt.step()
    pq = [par.item() for par in pars[:2]]
    plt.plot(pq[0], pq[1], ".")

pars = pars.detach()
pars[7:].clamp_(0)
print(pars)

plt.plot(np.linspace(0,1), np.ones(50)-np.linspace(0,1))
plt.show()

'''
# 2nd attempt
pars = torch.nn.Parameter(torch.randn(11))

# Calculates the norm symbolically with sympy
def LNorm(pars):
    p = \
    symbols('p q u1 u2 u3 u4 u5 s1 s2 s3 s4')
    p = [n for n in p]
    L = p[0]*p[1] + p[2]*(p[0]+p[1]-1) + p[3]*(-p[0]+p[7]) + p[4]*(p[0]-1+p[8])\
        + p[5]*(-p[1]+p[9]) + p[6]*(p[1]-1+p[10])
    grad = tensor.array.derive_by_array(L, p)
    Lnorm = sum([g**2 for g in grad])
    normGrad = tensor.array.derive_by_array(Lnorm, p)
    return normGrad.subs([(p[i], pars[i]) for i in range(len(p))])
±+
for epoch in range(1000):
    if epoch%100 == 0:
        print(epoch)
    opt.zero_grad()
    print(pars.grad)
    grads = LNorm(pars)
    with torch.no_grad():
        for i in range(len(pars)):
            new_val = pars[i].item() - grads[i]*0.1
            pars[i].copy_(torch.tensor(new_val))
    pq = [par.item() for par in pars[:2]]
    plt.plot(pq[0], pq[1], ".")

pars = pars.detach()
pars[7:].clamp_(0)
print(pars)

plt.plot(np.linspace(0,1), np.ones(50)-np.linspace(0,1))
plt.show()
'''