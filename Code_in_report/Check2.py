import torch
import pytorch_measure as pm
import numpy as np
from tqdm import tqdm
import json 
import math



# Linear regression model
def regression_model(x,list):
     return list[0]*x+list[1]

#Load data of parameters from the file where you ran data_generator.py
param=np.load(f'../Finalized/test_data/params.npy')

for length in [100,500,1000]:
     success=[]
     tid=[]
     epoch=[]
     measures=[]
     for i in tqdm(range(50)):
          data=np.load(f'../Finalized/test_data/data_{length}_y_lin_{i}.npy')
          y=torch.from_numpy(data)
          x = torch.linspace(-5, 5, length)

          M=length #Amount of datapoints

          s=2 #Decides width of the measures
          aU=math.ceil(param[0][3*i+1]+s*param[1][3*i+1])
          aL=math.floor(param[0][3*i+1]-s*param[1][3*i+1])
          bU=math.ceil(param[0][3*i]+s*param[1][3*i])
          bL=math.floor(param[0][3*i]-s*param[1][3*i])
          Nb=2*(bU-bL)+1
          Na=2*(aU-aL)+1

          #Creates the measures
          a = pm.Measure(torch.linspace(aL, aU, Na), torch.ones(Na).double() / Na)
          b = pm.Measure(torch.linspace(bL, bU, Nb), torch.ones(Nb).double() / Nb)
          measure= [a,b]

          # Instance of optimizer
          opt = pm.Optimizer(measure, "KDEnll", lr = 0.1)

          # Call to minimizer
          new_mes,time,iteration=opt.minimize([x,y],regression_model,max_epochs=4000,verbose = False, print_freq=100, smallest_lr=1e-10,test=True)

          # Visualize measures and gradient
          new_mes[0].visualize()
          #plt.show()
          new_mes[1].visualize()
          #plt.show()

          #Run tests and save the results
          check=pm.Check(opt,regression_model,x,y,normal=False,Return=True)
          l,u,miss=check.check()

          success.append(l<=miss and miss<=u)
          tid.append(time)
          epoch.append(iteration)
          for i in range(len(new_mes)):
               measures.append([new_mes[i].locations.tolist(),new_mes[i].weights.tolist()])

     data=[measures,sum(tid)/len(tid),sum(epoch)/len(epoch),sum(success)/len(success)]
     with open(f"resultat_samuel/Sergey2M_{M}.json", "w") as outfile:
          outfile.write(json.dumps(data))

