import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

N = 10
x = torch.linspace(3, 5, N)
y = 1.2 * x + torch.randn(N)

plt.scatter(x, y)
plt.show()

M = 50 # <- number of locations on measure

measure = pm.Measure(torch.linspace(0, 2, M), torch.ones(M) / M)

def error(x, a, y): # a is location in measure (scalar), for example slope in linear regression
    return ((a * x - y).pow(2)).sum()

def loss_fn(measure):
    errors = torch.tensor([error(x, measure.locations[j], y) for j in range(M)])
    return torch.dot(errors, measure.weights)

opt = pm.Optimizer(measure)
opt.minimize(loss_fn)

measure.visualize()

#plt.scatter(measure.locations.detach().numpy(), measure.weights.detach().numpy())
#plt.show()
