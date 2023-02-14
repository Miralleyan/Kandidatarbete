import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
mu=0
sigma=1
N=200
x=torch.linspace(-10,10,N)
#print(x)

y=1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-(x-mu)**2/(2*sigma**2))
y/=sum(y)
#print(y)

def loss_fn(w):
    x_d=torch.nn.parameter.Parameter(x)
    y_d=torch.nn.parameter.Parameter(y)
    m=mu
    s=sigma
    #return -sum(-1/(2*s**2)*((y_d-torch.mul(x_d,w))**2))
    return sum((y-w)**2)



def test_step():
    w = torch.tensor([1/N]*N)
    l = x
    measure = pm.PytorchMeasure(l, w)
    for epoch in range(2000):
        measure.step(loss_fn, 0.001)
        #print(measure)
    plt.scatter(x,y)
    measure.visualize()

test_step()