import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(10)
N = 10
x = torch.linspace(3, 5, N)
y = 1 * x + 1.5*torch.randn(N)

plt.scatter(x, y)
plt.show()

M = 100 # <- number of locations on measure

a = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)

def error(x, a, b, y): # a is location in measure (scalar), for example slope in linear regression
    return ((a * x + b - y).pow(2)).sum()

def loss_fn(measures):
    errors = torch.tensor([error(x, measures[0].locations[j], measures[1].locations[j], y) for j in range(M)])
    return torch.dot(errors, measures[0].weights) + torch.dot(errors, measures[1].weights)

opt = pm.Optimizer([a, b], lr = 0.5)
opt.minimize(loss_fn, verbose = True)

a.visualize()
b.visualize()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
plt.scatter(x,y)
plt.plot([3,5], [3*aMax+bMax, 5*aMax+bMax])
plt.show()
print(a.weights.grad)
print(b.weights.grad)