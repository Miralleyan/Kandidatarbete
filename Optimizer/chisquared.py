import scipy as sp
import pytorch_measure as pm
import torch

l = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1.])
w = torch.tensor([0.3, 0.15, 0.05, 0.05, 0.15, 0.3])
measure = pm.Measure(l, w)

data = torch.randn(1000)

def chi_squared(measure):
    bins = torch.tensor([torch.argmin(abs(data[i] - measure[0].weights)) for i in range(len(data))])
    bins_freq = []
    print(measure[0.weights])
    return sum(measure[0].weights**2/bins_freq)

opt = pm.Optimizer(measure, lr=0.01)
opt.minimize(chi_squared, verbose = True)
measure.visualize()