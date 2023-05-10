import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(30)
N = 200
x = torch.linspace(-1, 3, N)
y = (torch.randn(N)+1) * x + 0.1*torch.randn(N)

M = 30 # <- number of locations on measure

a = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)

def error(x, param, y): # a is location in measure (scalar), for example slope in linear regression
    return ((param[0] * x + param[1] - y).pow(2)).sum()

def loss_fn(measures: list[pm.Measure], n_samples=1000):
    #locs, probs = unif_samples(measures, n_samples)
    #errors = torch.tensor([error(x, locs[i], y) for i in range(n_samples)])
    #return errors.dot(probs)
    idx = torch.tensor([[i, j] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations))])
    locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    errors = torch.tensor([error(x, locs[i], y) for i in range(len(idx))])
    return errors.dot(probs)


opt = pm.Optimizer([a, b], lr = 0.001)
opt.minimize(loss_fn, max_epochs=1000, verbose = True, print_freq=5)

a.visualize()
b.visualize()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
plt.scatter(x,y)
plt.plot([-1,2], [-aMax+bMax, 2*aMax+bMax])
plt.show()
print(a.weights.grad)
print(b.weights.grad)