import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#torch.manual_seed(30) # <-- if seed is wanted
N = 100
x = torch.linspace(-3, 5, N)
 

# Plot the data points
#plt.scatter(x, y)
plt.show()

# Number of locations of measure
M = 17



# Linear regression model
def regression_model(x,list):
     return list[0]*x**2+list[1]*x+list[2]



success=[]
tid=[]
epoch=[]
for i in tqdm(range(50)):
    # Measure for slope (a) and intercept (b) of linear model
    a = pm.Measure(torch.linspace(0, 8, M), torch.ones(M) / M)
    b = pm.Measure(torch.linspace(-4, 4, M), torch.ones(M) / M)
    c= pm.Measure(torch.linspace(-2, 6, M), torch.ones(M) / M)


    measures = [a,b,c]

    y = (torch.randn(N)+4)*x**2+(torch.randn(N)+-0.5) * x + (2+torch.randn(N))
    # Instance of optimizer
    opt = pm.Optimizer(measures, "KDEnll", lr = 0.1)
    # Call to minimizer
    new_mes,time,iteration=opt.minimize([x,y],regression_model,max_epochs=3000,verbose = False, print_freq=100, smallest_lr=1e-10)
    # Visualize measures and gradient
    new_mes[0].visualize()
    #plt.show()
    new_mes[1].visualize()
    #plt.show()
    new_mes[2].visualize()
    #plt.show()

    check=pm.Check(opt,regression_model,x,y,normal=True,Return=True)
    l,u,miss=check.check()
    #check.check()
    success.append(l<=miss and miss<=u)
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