import sys
import torch
import pytorch_measure as pm
import numpy as np
import math

l = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1.])
w = torch.tensor([0.3, 0.15, 0.05, 0.05, 0.15, 0.3])
measure = pm.PytorchMeasure(l, w)

data = [0., 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 1.]

def loss_fn(w):
    return -sum([torch.log(w[torch.nonzero(l==data[i]).item()]) for i in range(len(data))])

print(loss_fn(measure.weights))
measure.vizualize()
for epoch in range(1000):
    measure.step(loss_fn, 0.05)
measure.vizualize()