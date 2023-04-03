import torch
import pytorch_measure as pm
import matplotlib.pyplot as plt
import numpy as np

#torch.manual_seed(30)
N = 1000
x = torch.linspace(-4, 4, N)
y = (torch.randn(N)+-4) * x**2 + (-2+torch.randn(N)) * x + (3+torch.randn(N))

plt.scatter(x, y)
plt.show()

M = 50 # <- number of locations on measure

# Measures to optimize
a = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
c = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)

# Regression model
def regression_model(param, x):
     return param[0]*x**2+param[1]*x+param[2]

# Distance from model prediction to true data
def error(x, param, y): # a is location in measure (scalar), for example slope in linear regression
    return ((param[0] * x**2 + param[1] * x + param[2] - y).pow(2)).sum()

# Indexing of the measures and computation of errors for each triplet
measures = [a,b,c]
idx = torch.tensor([[i, j, k] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations)) for k in range(len(measures[2].locations))])
locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
errors = torch.tensor([error(x, locs[i], y) for i in range(len(idx))])

# MSE
def loss_fn(measures: list[pm.Measure], n_samples=1000):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return errors.dot(probs)

# Kernel
def K(d):
        return 1/np.sqrt(2*np.pi)*np.exp(-d**2/2)
h=1.06*N**(-1/5)
kde_mat = K((y.view(-1,1) - regression_model(locs.transpose(0,1), x.view(-1,1))) / h)

def KDElog_loss(measures):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return -(torch.matmul(kde_mat, probs) / (N*h)).log().sum()

def log_prep(x, locs, y):
    loc_idx = []
    for i in range(len(x)):
        ab = torch.abs(locs[:,0]*x[i]**2+locs[:,1]*x[i]+locs[:,2]-y[i])
        loc_idx.append(torch.argmin(ab))
    return torch.tensor(loc_idx)

loc_index = log_prep(x,locs,y)
def log_loss(measures: list[pm.Measure]):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return -sum(torch.log(probs[loc_index]))

# Prep for chi squared
alpha = 0.001
bins = log_prep(x, locs, y)
bins_freq = torch.bincount(bins, minlength=M**3)/len(x)**2
bins_freq = bins_freq*(1-alpha)+alpha / len(bins_freq)

# Chi squared loss
def chi_squared(measures: list[pm.Measure]):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return sum(probs**2/bins_freq)

opt = pm.Optimizer([a, b, c], lr = 0.05)
measures = opt.minimize(KDElog_loss, max_epochs=600, verbose = True, print_freq=1, smallest_lr=1e-7)
a,b,c = measures[0],measures[1],measures[2]
opt.visualize()
plt.show()
aMax = torch.sum(a.locations*a.weights).detach()
aVar = torch.sum(a.weights*(a.locations-aMax)**2).detach()
bMax = torch.sum(b.locations*b.weights).detach()
cMax = torch.sum(c.locations*c.weights).detach()
plt.scatter(x,y)
plt.plot(x, aMax*x**2+bMax*x+cMax, 'r-')
plt.plot(x, (aMax-aVar**0.5)*x**2+bMax*x+cMax, 'r--')
plt.plot(x, (aMax+aVar**0.5)*x**2+bMax*x+cMax, 'r--')
plt.show()
print(aMax)
print(bMax)
print(cMax)
print(c.weights.grad)