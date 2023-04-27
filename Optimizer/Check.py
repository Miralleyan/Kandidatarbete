import matplotlib.pyplot as plt
import torch
import pytorch_measure as pm
import numpy as np

amin = -7
amax = 3
N = 2*(amax-amin)+1 # number of atoms
M = 1000 # Number of datapoints
verbose = True
dev = 'cpu'



def regression_model(x,list):
    return list[0]*x+1

success=[]
for i in range(100):
    x = torch.linspace(0, 10, M)
    data = (torch.randn(M).to(dev) - 2)*x+1
    w = torch.rand(N,dtype=torch.float).to(dev)
    w = torch.nn.parameter.Parameter(w/w.sum())
    l = torch.linspace(amin, amax, N, requires_grad=False).to(dev)

    measure = pm.Measure(locations=l, weights=w, device=dev)

    opt_NLL = pm.Optimizer([measure],"KDEnll" ,lr=1e-1)
    new_mes=opt_NLL.minimize([x,data], regression_model,verbose=True,adaptive=False,max_epochs=2000)

    new_mes[0].visualize()
    #plt.show()
    check=pm.Check(opt_NLL,regression_model,x,data,normal=True,Return=True)
    l,u,miss=check.check()
    #check.check()
    success.append(l<miss and miss<u)

print(sum(success))


