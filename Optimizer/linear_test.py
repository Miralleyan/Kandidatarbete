import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
mu=0
sigma=1
N=200
x=torch.linspace(-10,10,N)
y=1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-(x-mu)**2/(2*sigma**2))#"Observed" data
y/=sum(y) #Normalize


def loss_fn(w):
    x_d=torch.nn.parameter.Parameter(x)
    y_d=torch.nn.parameter.Parameter(y)
    m=mu
    s=sigma
    #return -sum(-1/(2*s**2)*((y_d-torch.mul(x_d,w))**2))
    return sum((y-w)**2)/len(w)



def test_step():
    w = torch.tensor([1/N]*N)
    l = x
    measure = pm.PytorchMeasure(l, w)
    for epoch in range(2000):
        measure.step(loss_fn, 0.001)
        #print(measure)
    plt.scatter(x,y,zorder=2)
    print(measure.total_mass())
    d=measure.sample(1)
    print(d)
    measure.visualize()


test_step()