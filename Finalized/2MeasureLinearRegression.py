import torch
import PyTorchMeasure as pm
import numpy as np
import matplotlib.pyplot as plt

#torch.manual_seed(30) # <-- if seed is wanted
N = 1000
x = torch.linspace(-1, 2, N)
y = (torch.randn(N)+-0.5) * x + (2+torch.randn(N))

# Plot the data points
plt.scatter(x, y)
plt.show()

# Number of locations of measure
M = 50

# Measure for slope (a) and intercept (b) of linear model
a = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-1, 3, M), torch.ones(M) / M)

# Linear regression model
def regression_model(param, x):
     return param[0]*x+param[1]

# Calculate the error of the model for a chosen pair of parameters and a point
def error(x, param, y):
    return ((param[0] * x + param[1] - y).pow(2)).sum()

# Indexing of measures into pairs and computation of errors for each pair
measures = [a,b]
idx = torch.tensor([[i, j] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations))])
locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
errors = torch.tensor([error(x, locs[i], y) for i in range(len(idx))])

# Calculate the expected sum of square residuals
# Degenerates to a good solution
def essr(measures: list[pm.Measure]):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return errors.dot(probs)

# Prepare data before calculation of negative log-likelihood
def log_prep(x, locs, y):
    loc_idx = []
    for i in range(len(x)):
        ab = torch.abs(locs[:,0]*x[i]+locs[:,1]-y[i])
        loc_idx.append(torch.argmin(ab))
    return torch.tensor(loc_idx)

# Calculate index of closest location to each data point
loc_index = log_prep(x, locs, y)
# Calulate the negative log-likelihood
def log_loss(measures: list[pm.Measure]):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return -(probs[loc_index].log()).sum()

# Kernel for KDE
def K(d):
        return 1/np.sqrt(2*np.pi)*np.exp(-d**2/2)
# Bandwidth parameter
h=1.06*N**(-1/5)
# KDE
kde_mat = K((y.view(-1,1) - regression_model(locs.transpose(0,1), x.view(-1,1))) / h)

# Negative log-likelihood with KDE of measures
def KDElog_loss(measures):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return -(torch.matmul(kde_mat, probs) / (N*h)).log().sum()

# Prepare data before calculation of chi-squared-loss
def indexing(measures: list[pm.Measure]):
    idx = torch.tensor([[i, j] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations))])
    locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return locs, probs

# Alpha parameter for label smoothing in chi-squared-loss
alpha = 0.001
# Calculate chi-squared-loss
def chi_squared(measures: list[pm.Measure]):
    locs, probs = indexing(measures)
    bins = log_prep(x, locs, y)
    bins_freq = torch.bincount(bins, minlength=900)/len(x)**2
    bins_freq = bins_freq*(1-alpha)+alpha / len(bins_freq)
    return sum(probs**2/bins_freq)

# Instance of optimizer
opt = pm.Optimizer([a, b], lr = 0.05)
# Call to miinimizer
opt.minimize(KDElog_loss, max_epochs=1000, verbose = True, print_freq=100, smallest_lr=1e-10)
# Visualize measures and gradient
opt.visualize()
plt.show()
# Calculate mean of the measures
aMean = torch.sum(a.locations*a.weights).detach()
bMean = torch.sum(b.locations*b.weights).detach()
# Plot the data along with a linear model using the means as parameters
plt.scatter(x,y)
plt.plot([-1,2], [-aMean+bMean, 2*aMean+bMean],'r-')
plt.show()
print(f'Model: y = {aMean}x+{bMean}')