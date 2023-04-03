import torch
import pytorch_measure as pm
import matplotlib.pyplot as plt

#torch.manual_seed(30)
N = 200
x = torch.linspace(-2, 2, N)
y = (torch.randn(N)+1.5) * x**2 + (1+torch.randn(N)) * x + 2*(0.5+torch.randn(N))

plt.scatter(x, y)
plt.show()

M = 30 # <- number of locations on measure

a = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)
c = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)

def error(x, param, y): # a is location in measure (scalar), for example slope in linear regression
    return ((param[0] * x**2 + param[1] * x + param[2] - y).pow(2)).sum()
    # estimate = sum([par * x**(len(param)-i-1) for i, par in enumerate(param)])
    # return ((estimate - y).pow(2)).sum()

measures = [a,b,c]
idx = torch.tensor([[i, j, k] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations)) for k in range(len(measures[2].locations))])
locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
errors = torch.tensor([error(x, locs[i], y) for i in range(len(idx))])
def loss_fn(measures: list[pm.Measure], n_samples=1000):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return errors.dot(probs)


opt = pm.Optimizer([a, b, c], lr = 0.01)
opt.minimize(loss_fn, max_epochs=1000, verbose = True, print_freq=5)
opt.visualize()
plt.show()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
cMax = c.locations[torch.argmax(c.weights)]
plt.scatter(x,y)
plt.plot(x, aMax*x**2+bMax*x+cMax)
plt.show()