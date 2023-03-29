'''
This script tests linear regression with the model
2.15x+0.5*alpha, where alpha is a normally distributed
variable
'''
import torch
import PyTorchMeasure as pm
import matplotlib.pyplot as plt

N = 100
x = torch.linspace(0, 5, N)
y = 2.15 * x + 0.5 * torch.randn(N)

plt.scatter(x, y)
plt.show()

M = 1000 # <- number of locations on measure

measure = pm.Measure(torch.linspace(0, 5, M), torch.ones(M) / M)

def error(x, a, y): # a is location in measure (scalar), for example slope in linear regression
    return ((a * x - y).pow(2)).sum()

def loss_fn(measures):
    errors = torch.tensor([error(x, measures[0].locations[j], y) for j in range(M)])
    return torch.dot(errors, measures[0].weights)

opt = pm.Optimizer(measure, lr = 1.0)
opt.minimize(loss_fn, verbose = True)
print(measure.locations[torch.argmax(measure.weights)].item())  # Print degenerated solution

measure.visualize()
opt.visualize()
plt.show()