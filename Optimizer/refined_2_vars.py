# Based on 2_vars_test.py
# This version aims to speed up the calculation by not computing ESSR for each
# combination of weights, but instead calculating for each combination with index
# divisible by ref_par

import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(15)
N = 100
x = torch.linspace(-1, 2, N)
y = 2 * x + 1 + 0.5*torch.randn(N)

#plt.scatter(x, y)
#plt.show()

M = 100 # <- number of locations on measure

a = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)

def error(x, param, y): # a is location in measure (scalar), for example slope in linear regression
    return ((param[0] * x + param[1] - y).pow(2)).sum()

ref_par = 3

def loss_fn(measures: list[pm.Measure]):
    #locs, probs = unif_samples(measures, n_samples)
    #errors = torch.tensor([error(x, locs[i], y) for i in range(n_samples)])
    #return errors.dot(probs)
    idx = torch.tensor([[i, j] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations))])
    idx = idx[::ref_par]
    locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    errors = torch.tensor([error(x, locs[i], y) for i in range(len(idx))])
    return errors.dot(probs)

opt = pm.Optimizer([a, b], lr = 0.01)
opt.minimize(loss_fn, max_epochs=100, verbose = True, print_freq=5)

a.visualize()
b.visualize()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
plt.scatter(x,y)
plt.plot([-1,2], [-aMax+bMax, 2*aMax+bMax])
mu_a = sum(a.weights*a.locations)
mu_b = sum(b.weights*b.locations)
var_a = sum(a.weights*(a.locations-mu_a)**2).item()
print(var_a)
var_b = sum(b.weights*(b.locations-mu_b)**2).item()
#plt.plot([-1,2], [-(aMax)+bMax-var_b, 2*(aMax)+bMax-var_b], 'r--')
#plt.plot([-1,2], [-(aMax)+bMax+var_b, 2*(aMax)+bMax+var_b], 'r--')
plt.show()
print(a.weights.grad)
print(b.weights.grad)