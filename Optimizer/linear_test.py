import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

N=200
x=torch.randn(N)


mu=0
sigma=1
xt=torch.linspace(-10,10,N)
yt=1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-(xt-mu)**2/(2*sigma**2))
yt/=sum(yt) #Normalize


w = torch.tensor([1/N]*N)#Weights
l = xt#Locations
measure = pm.PytorchMeasure(l, w)


def loss_fn(w):
    return sum((yt-w)**2)/len(w)
    #return -sum([torch.log(w[torch.nonzero(l==x_d[i]).item()]) for i in range(len(x_d))])



for epoch in range(2000):
    measure.step(loss_fn, 0.001)
    #print(measure)
plt.scatter(xt,yt,zorder=2)
print(1-measure.total_mass())
measure.visualize()


