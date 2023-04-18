import matplotlib.pyplot as plt
import torch
import pytorch_measure as pm
import numpy as np
N = 17 # number of atoms
M = 4000 # Number of datapoints
amin = -5
amax = 3
verbose = True
dev = 'cpu'




def regression_model(x,list):
    return list[0]+x


x = torch.linspace(0, 10, M).view(-1, 1)
data = regression_model(torch.randn(M).to(dev) - 2, x.view(1, -1)).view(-1, 1)
w = torch.rand(N,dtype=torch.float).to(dev)
w = torch.nn.parameter.Parameter(w/w.sum())
l = torch.linspace(amin, amax, N, requires_grad=False).to(dev)

measure = pm.Measure(locations=l, weights=w, device=dev)

opt_NLL = pm.Optimizer([measure],"nll" ,lr=1e-1)
new_mes=opt_NLL.minimize([x,data], regression_model,verbose=True)

new_mes[0].visualize()
plt.show()
check=pm.Check(opt_NLL,regression_model,x,data)
prob,miss=check.check()
print(prob)
print(miss)
