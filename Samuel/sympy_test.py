from sympy import *
import torch

def LNorm(pars):
    p = \
    symbols('p q u1 u2 u3 u4 u5 s1 s2 s3 s4')
    p = [n for n in p]
    print(p)
    L = p[0]*p[1] + p[2]*(p[0]+p[1]-1) + p[3]*(-p[0]+p[7]) + p[4]*(p[0]-1+p[8])\
        + p[5]*(-p[1]+p[9]) + p[6]*(p[1]-1+p[10])
    grad = tensor.array.derive_by_array(L, p)
    Lnorm = sum([g**2 for g in grad])
    print(type(p))
    normGrad = tensor.array.derive_by_array(Lnorm, p)
    return normGrad.subs([(p[i], pars[i]) for i in range(len(p))])

pars = torch.nn.Parameter(torch.randn(11))
print(LNorm(pars))
