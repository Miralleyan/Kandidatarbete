import torch
import pytorch_measure as pm
import numpy as np
mu=0
sigma=1
x=np.linspace(-3,3,100)
#print(x)

y=1/(2*np.pi*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
#print(y)

def loss_fn():
    return -sum(-1/(2))

def test_step():
    w = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    l = torch.tensor([-2.,-1.,0.,1.,2.])
    measure = pm.Pytorch_measure(l, w)
    for epoch in range(100):
        measure.step(loss_fn, 0.01)
        print(measure)

test_step()