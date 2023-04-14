import torch
import pytorch_measure as pm
import matplotlib.pyplot as plt

#torch.manual_seed(30)
N = 1000
x = torch.linspace(-1, 2, N)
y = (torch.randn(N)+-0.5) * x + (2+torch.randn(N))

M = 30 # <- number of locations on measure

a = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)

def model(x, params):
    return params[0]*x+params[1]

opt = pm.Optimizer([a, b], 'nll', lr = 0.05)
opt.minimize([x,y], model, max_epochs=500, verbose = True, print_freq=5)

opt.visualize()
plt.show()
opt.measures[0].visualize()
plt.show()
opt.measures[1].visualize()
plt.show()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
plt.scatter(x,y)
plt.plot([-1,2], [-aMax+bMax, 2*aMax+bMax],'r-')
plt.show()
print(a.weights.grad)
print(b.weights.grad)