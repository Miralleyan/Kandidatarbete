# File for comparing results of three measure quadratic
# regression between methods

import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import linear_combination_optimizer as lco
from tqdm import tqdm
import json

# Confidence intervals
def misses(x, y, mu, sigma):
    miss = 0
    S = 1000
    for i in range(x.size(dim=0)):
        sample = torch.normal(mean = float(mu[i]), std = float(sigma[i]), size = (1,S)).squeeze(dim=0)
        sample = torch.sort(sample)[0]
        if y[i] < sample[int(np.floor(S*0.025))] or y[i] > sample[int(np.ceil(S*0.975))-1]:
            miss += 1
    c1 = sp.stats.binom.ppf(0.025,y.size(dim=0),0.05)
    c3 = sp.stats.binom.ppf(0.5,y.size(dim=0),0.05)
    c2 = sp.stats.binom.ppf(0.975,y.size(dim=0),0.05)
    print(c1,c3,c2)
    print(f"CI: [{c1}, {c2}], Misses: {miss}, Within CI: {c1<=miss<=c2}")
    return c1, c2, miss

for length in [100, 500, 1000]:
    success=[]
    time=[]
    epoch=[]
    means=[]
    std=[]
    for i in range(50):
        x = torch.linspace(-5, 5, length)
        y = np.load(f'../Finalized/test_data/data_{length}_y_sqr_{i}.npy')
        opt = lco.Optimizer(x, y, order=3)
        mu, sigma, conv_epoch, conv_time = opt.optimize(epochs = 3000, test = True)

        l, u, miss = misses(x,torch.tensor(y),mu,sigma)
        success.append(l<=miss and miss<=u)
        time.append(conv_time)
        epoch.append(conv_epoch)
        means.append(mu)
        std.append(sigma)
    
    results = [means, std, sum(time)/len(time), sum(epoch)/(len(epoch)), float(100*sum(success)/len(success))]
    with open(f"Lin_comb_results/lin_comb_results_{length}_y_sqr.json", "w") as outfile:
        outfile.write(json.dumps(results))