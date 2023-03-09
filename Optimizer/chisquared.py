import scipy as sp
import pytorch_measure as pm
import torch
from matplotlib import pyplot as plt

l = torch.linspace(-3, 3, 20)
w = torch.ones(20) / 20
measure = pm.Measure(l, w)

data = torch.randn(10000)
alpha = 0.01


def chi_squared(measure):
    # Calculate frequencies of data
    bins = torch.tensor([torch.argmin(abs(data[i] - measure[0].locations)) for i in range(len(data))])
    bins_freq = bins.unique(return_counts=True)[1] / len(data)
    # Label smoothing
    bins_freq = bins_freq*(1-alpha)+alpha / len(bins_freq)
    return 100 * sum(measure[0].weights**2 / bins_freq)


opt = pm.Optimizer(measure, lr=0.01)
opt.minimize(chi_squared, smallest_lr=1e-3, verbose = True, print_freq=1)
measure.visualize()
