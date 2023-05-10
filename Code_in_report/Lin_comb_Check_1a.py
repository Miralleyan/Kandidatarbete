import torch
import numpy as np
import scipy as sp
import linear_combination_optimizer as lco
import json

# Confidence intervals
def misses(x, y, mu, sigma):
    miss = 0
    S = 1000
    for i in range(x.size(dim=0)):
        # Generate sample from model
        sample = torch.normal(mean = float(mu[i]), std = float(sigma[i]), size = (1,S)).squeeze(dim=0)
        sample = torch.sort(sample)[0]
        if y[i] < sample[int(np.floor(S*0.025))] or y[i] > sample[int(np.ceil(S*0.975))-1]:
            miss += 1    
    # Calculate confidence interval
    c1 = sp.stats.binom.ppf(0.025,y.size(dim=0),0.05)
    c2 = sp.stats.binom.ppf(0.975,y.size(dim=0),0.05)
    print(f"CI: [{c1}, {c2}], Misses: {miss}, Within CI: {c1<=miss<=c2}")
    return c1, c2, miss

# For each data size
for length in [100, 500, 1000]:
    # Lists for storing results
    success=[]
    time=[]
    epoch=[]
    means=[]
    std=[]
    # Run regressions 50 times
    for i in range(50):
        x = torch.linspace(-5, 5, length)
        y = np.load(f'../Finalized/test_data/data_{length}_y_ax_{i}.npy')
        opt = lco.Optimizer(x, y, order=1, ax = True)
        mu, sigma, conv_epoch, conv_time = opt.optimize(epochs = 3000, test = True)

        l, u, miss = misses(x,torch.tensor(y),mu,sigma)
        success.append(l<=miss and miss<=u)
        time.append(conv_time)
        epoch.append(conv_epoch)
        means.append(mu)
        std.append(sigma)
    
    # Save results in a json file
    results = [means, std, sum(time)/len(time), sum(epoch)/(len(epoch)), float(100*sum(success)/len(success))]
    with open(f"Lin_comb_results/lin_comb_results_{length}_y_ax.json", "w") as outfile:
        outfile.write(json.dumps(results))