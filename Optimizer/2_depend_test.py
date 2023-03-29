import torch
import pytorch_measure as pm
import matplotlib.pyplot as plt

N = 100
x = torch.linspace(0, 5, N)
y = (1 + torch.randn(N)) * x + 0.5 * torch.randn(N)

plt.scatter(x, y)
plt.show()

M = 100 # <- number of locations on measure
loc = torch.linspace(0, 5, M)
locs = []
for i in range(M):
    for j in range(M):
        locs.append((loc[i].item(),loc[j].item()))
locs = torch.tensor(locs)
print(locs.size(dim=0))
measure = pm.Measure(locs, torch.ones(M**2) / M**2)

def error(x, a, b, y): # a is location in measure (scalar), for example slope in linear regression
    return ((a * x + b - y).pow(2)).sum()

def loss_fn(measures):
    errors = torch.tensor([error(x, measures[0].locations[j][0], measures[0].locations[j][1], y) for j in range(M**2)])
    return torch.dot(errors, measures[0].weights)

opt = pm.Optimizer(measure, lr = 1.0)
opt.minimize(loss_fn, verbose = True)
loc = torch.meshgrid(loc)
plt.plot()
plt.show()