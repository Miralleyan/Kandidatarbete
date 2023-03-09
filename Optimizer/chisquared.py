import scipy as sp
import pytorch_measure as pm
import torch
from matplotlib import pyplot as plt

l = torch.tensor([-1., -0.6, -0.2, 0.2, 0.6, 1.])
w = torch.tensor([0.3, 0.15, 0.05, 0.05, 0.15, 0.3])
measure = pm.Measure(l, w)

data = torch.randn(100000)
alpha = 0.01
plt.hist(data)
plt.show()

def chi_squared(measure):
    # Calculate frequencies of data
    bins = torch.tensor([torch.argmin(abs(data[i] - measure[0].locations)) for i in range(len(data))])
    bins_freq = bins.unique(return_counts=True)[1]/len(data)
    # Label smoothing
    bins_freq = bins_freq*(1-alpha)+alpha/len(bins_freq)
    print(bins_freq)

    return sum(measure[0].weights**2/bins_freq)

opt = pm.Optimizer(measure, lr=0.01)
opt.minimize(chi_squared, verbose = True)
measure.visualize()