import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(30)
N = 200
x = torch.linspace(-1, 3, N)
y = (torch.randn(N)+1) * x + 0.1*torch.randn(N)

#plt.scatter(x, y)
#plt.show()

M = 30 # <- number of locations on measure

a = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)

def error(x, param, y): # a is location in measure (scalar), for example slope in linear regression
    return ((param[0] * x + param[1] - y).pow(2)).sum()

# return list[(sample (tensor -- list of locations), tensor -- probability)]
# assumes that the variables are independent
def unif_samples(ms: list[pm.Measure], n_samples):
    idx = (torch.rand((n_samples, len(ms))) * torch.tensor([len(m.locations) for m in ms])).long()
    locs = torch.cat([ms[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1)
    probs = torch.cat([ms[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1).prod(1)
    return (locs, probs)

def sample_meas(ms: list[pm.Measure], n_samples):
    idx = torch.cat([ms[i].sample(n_samples).unsqueeze(1) for i in range(len(ms))], 1).long()
    locs = torch.cat([ms[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1)
    probs = torch.cat([ms[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1).prod(1)
    return (locs, probs)

def loss_fn(measures: list[pm.Measure], n_samples=1000):
    #locs, probs = unif_samples(measures, n_samples)
    #errors = torch.tensor([error(x, locs[i], y) for i in range(n_samples)])
    #return errors.dot(probs)
    idx = torch.tensor([[i, j] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations))])
    locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    errors = torch.tensor([error(x, locs[i], y) for i in range(len(idx))])
    return errors.dot(probs)

def loss_fn_2(measures: list[pm.Measure], n_samples=1000):
    locs, probs = sample_meas(measures, n_samples)
    errors = torch.tensor([error(x, locs[i], y) for i in range(n_samples)])
    return errors.dot(probs)

def log_prep(x, locs, y):
    loc_idx = []
    for i in range(len(x)):
        ab = torch.abs(locs[:,0]*x[i]+locs[:,1]-y[i])
        loc_idx.append(torch.argmin(ab))
    return torch.tensor(loc_idx)

def log_loss(measures: list[pm.Measure]):
    idx = torch.tensor([[i, j] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations))])
    locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    loc_index = log_prep(x, locs, y)
    return -(probs[loc_index].log()).sum()

def indexing(measures: list[pm.Measure]):
    idx = torch.tensor([[i, j] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations))])
    locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return locs, probs

alpha = 0.001
def chi_squared(measures: list[pm.Measure]):
    locs, probs = indexing(measures)
    bins = log_prep(x, locs, y)
    bins_freq = torch.bincount(bins, minlength=900)/len(x)**2
    bins_freq = bins_freq*(1-alpha)+alpha / len(bins_freq)
    return sum(probs**2/bins_freq)


opt = pm.Optimizer([a, b], lr = 0.001)
opt.minimize(log_loss, max_epochs=1000, verbose = True, print_freq=5)
"""
a.visualize()
b.visualize()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
plt.scatter(x,y)
plt.plot([-1,2], [-aMax+bMax, 2*aMax+bMax])
"""
opt.visualize()
plt.show()
print(a.weights.grad)
print(b.weights.grad)