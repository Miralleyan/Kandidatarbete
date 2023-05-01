import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

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




for length in [100,500,1000]:

    data=np.load(f'../Finalized/test_data/data_{length}_y_lin.npy')
    x=torch.from_numpy(data.reshape(1,-2)[0][:length])
    y=torch.from_numpy(data.reshape(1,-2)[0][length:])
    x,y=torch.tensor(x.tolist()),torch.tensor(y.tolist())

    M=length #Amount of datapoints
    N=21




success=[]
tid=[]
epoch=[]
measures=[]
for i in tqdm(range(50)):

     # Measure for slope (a) and intercept (b) of linear model
     a = pm.Measure(torch.linspace(-4, 4, M), torch.ones(M) / M)
     b = pm.Measure(torch.linspace(-2, 6, M), torch.ones(M) / M)


     measure= [a,b]
     y = (torch.randn(N)+-0.5) * x + (2+torch.randn(N))
     # Instance of optimizer
     opt = pm.Optimizer(measure, "KDEnll", lr = 0.1)
     # Call to minimizer
     new_mes,time,iteration=opt.minimize([x,y],regression_model,max_epochs=2000,verbose = False, print_freq=100, smallest_lr=1e-10,test=True)
     # Visualize measures and gradient
     new_mes[0].visualize()
     #plt.show()
     new_mes[1].visualize()
     #plt.show()

     check=pm.Check(opt,regression_model,x,y,normal=True,Return=True)
     l,u,miss=check.check()
     #check.check()
     success.append(l<=miss and miss<=u)
     tid.append(time)
     epoch.append(iteration)
     for i in range(len(new_mes)):
          print(new_mes[i].locations)
          measures.append([new_mes[i].locations.tolist(),new_mes[i].weights.tolist()])

data=[measures,sum(tid)/len(tid),sum(epoch)/len(epoch),sum(success)/len(success)]
with open(f"Sergey2M:{N}.json", "w") as outfile:
    outfile.write(json.dumps(data))


print(sum(success)/len(success))
print(sum(tid)/len(tid))
print(sum(epoch)/len(epoch))