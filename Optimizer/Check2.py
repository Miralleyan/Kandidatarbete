import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import math

'''
#torch.manual_seed(30) # <-- if seed is wanted
N = 1000
x = torch.linspace(-3, 5, N)


# Plot the data points
#plt.scatter(x, y)
plt.show()

# Number of locations of measure
M = 17

'''


# Linear regression model
def regression_model(x,list):
     return list[0]*x+list[1]


param=np.load(f'../Finalized/test_data/params.npy')
for length in [1000]:
     success=[]
     tid=[]
     epoch=[]
     measures=[]
     for i in tqdm(range(50)):
          data=np.load(f'../Finalized/test_data/data_{length}_y_lin_{i}.npy')
          x=torch.from_numpy(data.reshape(1,-2)[0][:length])
          print(x)
          y=torch.from_numpy(data.reshape(1,-2)[0][length:])
          x,y=torch.tensor(x.tolist()),torch.tensor(y.tolist())

          M=length #Amount of datapoints

          s=2
          aU=math.ceil(param[0][3*i+1]+s*param[1][3*i+1])
          aL=math.floor(param[0][3*i+1]-s*param[1][3*i+1])
          bU=math.ceil(param[0][3*i]+s*param[1][3*i])
          bL=math.floor(param[0][3*i]-s*param[1][3*i])
          Nb=2*(bU-bL)+1
          Na=2*(aU-aL)+1
          N=max(Nb,Na)

          a = pm.Measure(torch.linspace(aL, aU, Na), torch.ones(Na) / Na)
          b = pm.Measure(torch.linspace(bL, bU, Nb), torch.ones(Nb) / Nb)


          measure= [a,b]
          #y = (torch.randn(M)+-0.5) * x + (2+torch.randn(M))
          # Instance of optimizer
          opt = pm.Optimizer(measure, "KDEnll", lr = 0.1)
          # Call to minimizer
          new_mes,time,iteration=opt.minimize([x,y],regression_model,max_epochs=3000,verbose = False, print_freq=100, smallest_lr=1e-10,test=True)
          opt.visualize()
          # Visualize measures and gradient
          new_mes[0].visualize()
          
          new_mes[1].visualize()
          plt.show()

          check=pm.Check(opt,regression_model,x,y,normal=False,Return=True)
          l,u,miss=check.check()
          success.append(l<=miss and miss<=u)
          tid.append(time)
          epoch.append(iteration)
          for i in range(len(new_mes)):
               measures.append([new_mes[i].locations.tolist(),new_mes[i].weights.tolist()])

     data=[measures,sum(tid)/len(tid),sum(epoch)/len(epoch),sum(success)/len(success)]
     with open(f"Sergey2M:{M}.json", "w") as outfile:
          outfile.write(json.dumps(data))


print(sum(success)/len(success))
print(sum(tid)/len(tid))
print(sum(epoch)/len(epoch))