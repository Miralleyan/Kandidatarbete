import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(10)
N = 20
x = torch.linspace(3, 5, N)
y = 0.3*torch.randn(N) + 2

#plt.scatter(x, y)
#plt.show()

M = 20 # <- number of locations on measure

a = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)

def error(x, param, y): # a is location in measure (scalar), for example slope in linear regression
    return ((param[0] * x + param[1] - y).pow(2)).sum()

# return list[(sample (tensor -- list of locations), tensor -- probability)]
# assumes that the variables are independent
def unif_samples(ms: list[pm.Measure], n_samples):
    idx = (torch.rand((n_samples, len(ms))) * torch.tensor([len(m.locations) for m in ms])).long()
    locs = torch.cat([ms[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1)
    probs = torch.cat([ms[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1).prod(1)
    return (locs, probs)

def loss_fn(measures: list[pm.Measure], n_samples=1000):
    locs, probs = unif_samples(measures, n_samples)
    errors = torch.tensor([error(x, locs[i], y) for i in range(n_samples)])
    return errors.dot(probs)

opt = pm.Optimizer([a, b], lr = 0.1)
opt.minimize(loss_fn, max_epochs=1000, verbose = True)

a.visualize()
b.visualize()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
plt.scatter(x,y)
plt.plot([3,5], [3*aMax+bMax, 5*aMax+bMax])
plt.show()
print(a.weights.grad)
print(b.weights.grad)