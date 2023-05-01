import matplotlib.pyplot as plt
import torch
import pytorch_measure as pm
import numpy as np
from tqdm import tqdm
import json 

amin = -7
amax = 3
N = 2*(amax-amin)+1 # number of atoms
M = 1000 # Number of datapoints
verbose = True
dev = 'cpu'



def regression_model(x,list):
    return list[0]


success=[]
tid=[]
epoch=[]
measures=[]
for i in tqdm(range(50)):
    x = torch.linspace(0, 10, M)
    data = torch.randn(M).to(dev)-2
    w = torch.rand(N,dtype=torch.float).to(dev)
    w = torch.nn.parameter.Parameter(w/w.sum())
    l = torch.linspace(amin, amax, N, requires_grad=False).to(dev)

    measure = pm.Measure(locations=l, weights=w, device=dev)

    opt_NLL = pm.Optimizer([measure],"KDEnll" ,lr=1e-1)
    new_mes,time,iteration=opt_NLL.minimize([x,data], regression_model,verbose=False,adaptive=False,max_epochs=2000,test=True)

    new_mes[0].visualize()
    plt.show()
    check=pm.Check(opt_NLL,regression_model,x,data,normal=True,Return=True)
    l,u,miss=check.check()
    #check.check()
    success.append(l<miss and miss<u)
    tid.append(time)
    epoch.append(iteration)
    measures.append([new_mes[0].locations.tolist(),new_mes[0].weights.tolist()])

data=[measures,sum(tid)/len(tid),sum(epoch)/(len(epoch)),sum(success)/len(success)]
with open(f"Sergey1M:{M}.json", "w") as outfile:
    outfile.write(json.dumps(data))

print(sum(success))
print(sum(success)/50)
print(sum(tid)/len(tid))
print(sum(epoch)/(len(epoch)))

