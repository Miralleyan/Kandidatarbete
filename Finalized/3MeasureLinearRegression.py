import torch
import PyTorchMeasure as pm
import matplotlib.pyplot as plt
import numpy as np

#torch.manual_seed(30) # <-- if seed is wanted
N = 1000

# Produce the data points
x = torch.linspace(-4, 4, N)
y = (torch.randn(N)+-4) * x**2 + (-2+torch.randn(N)) * x + (3+torch.randn(N))

# Plot the data points
plt.scatter(x, y)
plt.show()

# Number of locations on measure
M = 50

# Measures for a quadratic model
a = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
c = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)

# Quadratic regression model
def regression_model(param, x):
     return param[0]*x**2+param[1]*x+param[2]

# Distance from model prediction to true data
def error(x, param, y):
    return ((param[0] * x**2 + param[1] * x + param[2] - y).pow(2)).sum()

# Indexing of the measures into triplets and computation of errors for each triplet
measures = [a,b,c]
idx = torch.tensor([[i, j, k] for i in range(len(measures[0].locations)) for j in range(len(measures[1].locations)) for k in range(len(measures[2].locations))])
locs = torch.cat([measures[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1)
errors = torch.tensor([error(x, locs[i], y) for i in range(len(idx))])

# Calculate expected sum of square residuals
# Degenerates to a good solution
def essr(measures: list[pm.Measure]):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return errors.dot(probs)

# Prepare data before calculation of negative log-likelihood
def log_prep(x, locs, y):
    loc_idx = []
    for i in range(len(x)):
        ab = torch.abs(locs[:,0]*x[i]**2+locs[:,1]*x[i]+locs[:,2]-y[i])
        loc_idx.append(torch.argmin(ab))
    return torch.tensor(loc_idx)

# Calculate index of closest location to each data point
loc_index = log_prep(x,locs,y)
# Negative log-likelihood
def log_loss(measures: list[pm.Measure]):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return -sum(torch.log(probs[loc_index]))

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

# Alpha parameter for label smoothing in chi-squared-loss
alpha = 0.001
# Calculate amount of data points closest to a location and the frequencies
bins = log_prep(x, locs, y)
bins_freq = torch.bincount(bins, minlength=M**3)/len(x)**2
bins_freq = bins_freq*(1-alpha)+alpha / len(bins_freq)

# Chi squared loss
def chi_squared(measures: list[pm.Measure]):
    probs = torch.cat([measures[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(measures))], 1).prod(1)
    return sum(probs**2/bins_freq)

# Instance of optimizer
opt = pm.Optimizer([a, b, c], lr = 0.05)
# Call to minimizer
measures = opt.minimize(KDElog_loss, max_epochs=600, verbose = True, print_freq=1, smallest_lr=1e-7)
a,b,c = measures[0],measures[1],measures[2]
# Visualize measures and gradient
opt.visualize()
plt.show()
# Calculate mean of measures
aMean = torch.sum(a.locations*a.weights).detach()
bMean = torch.sum(b.locations*b.weights).detach()
cMean = torch.sum(c.locations*c.weights).detach()
# Plot the data along with a quadratic model using the means as parameters
plt.scatter(x,y)
plt.plot(x, aMean*x**2+bMean*x+cMean, 'r-')
plt.show()
print(f'Model: y = {aMean}x^2+{bMean}x+{cMean}')