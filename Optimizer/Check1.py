import matplotlib.pyplot as plt
import torch
import pytorch_measure as pm
import numpy as np
from tqdm import tqdm
import json 
import math
dev = 'cpu'


'''
amin = -3
amax = 3
N = 2*(amax-amin)+1 # number of atoms
M = 1000 # Number of datapoints
verbose = True
'''



def regression_model(x,list):
    return list[0]

param=np.load(f'../Finalized/test_data/params.npy')

for length in [100]:
    success=[]
    tid=[]
    epoch=[]
    measures=[]
    for i in tqdm(range(50)):
        data=np.load(f'../Finalized/test_data/data_{length}_y_{i}.npy')
        print(data[2])
        y=torch.from_numpy(data)
        x = torch.linspace(-5, 5, length)
        plt.scatter(x,y)
        plt.show()

        #y=(torch.randn(length)*param[1][3*i]+param[0][3*i]).double()
        M=length #Amount of datapoints

        s=2
        aU=math.ceil(param[0][3*i]+s*param[1][3*i])
        aL=math.floor(param[0][3*i]-s*param[1][3*i])
        N=2*(aU-aL)+1
        
 
        #x = torch.linspace(0, 10, M)
        #data = torch.randn(M).to(dev)
        measure = pm.Measure(torch.linspace(aL, aU, N), torch.ones(N).double() / N)
        #w = torch.rand(N,dtype=torch.float).to(dev)
        #w = torch.nn.parameter.Parameter(w/w.sum())
        #l = torch.linspace(-2, 3, N, requires_grad=False).to(dev)

        #measure = pm.Measure(locations=l, weights=w, device=dev)

        opt = pm.Optimizer([measure],"KDEnll" ,lr=1e-1)
        new_mes,time,iteration=opt.minimize([x,y], regression_model,verbose=False,adaptive=False,max_epochs=4000,test=True)

        new_mes[0].visualize()
        plt.show()
        check=pm.Check(opt,regression_model,x,y,normal=True,Return=True)
        l,u,miss=check.check()

        success.append(l<miss and miss<u)
        tid.append(time)
        epoch.append(iteration)
        measures.append([new_mes[0].locations.tolist(),new_mes[0].weights.tolist()])

    data=[measures,sum(tid)/len(tid),sum(epoch)/(len(epoch)),sum(success)/len(success)]
    #with open(f"resultat_sergey/Sergey1M_{M}.json", "w") as outfile:
    #    outfile.write(json.dumps(data))

print(sum(success)/len(success))
print(sum(tid)/len(tid))
print(sum(epoch)/(len(epoch)))


