import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

N=200
data=torch.randn(N)


mu=0
sigma=1
xt=torch.linspace(-10*sigma+mu,10*sigma+mu,N)
yt=1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-(xt-mu)**2/(2*sigma**2))
yt/=sum(yt) #Normalize


w = torch.tensor([1/N]*N)#Weights
l = xt#Locations
w = torch.nn.parameter.Parameter(w)
l = torch.nn.parameter.Parameter(l)

measure = pm.Measure(l, w)
opt=pm.Optimizer(measure)


def loss_fn(w):
    return sum((yt-w)**2)/len(w)
    #return -sum([torch.log(w[torch.nonzero(l==data[i].item()).item()]) for i in range(len(data))])



for epoch in range(2000):
    loss=loss_fn(w)
    loss.backward
    opt.step(lr=0.001)
    #print(measure)
plt.scatter(xt,yt,zorder=2)
print(1-measure.total_mass())
measure.visualize()


