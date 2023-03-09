import scipy as sp
import pytorch_measure as pm
import torch

l = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1.])
w = torch.tensor([0.3, 0.15, 0.05, 0.05, 0.15, 0.3])
measure = pm.Measure(l, w)

