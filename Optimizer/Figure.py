import matplotlib.pyplot as plt
import numpy as np
import torch
import linear_combination_optimizer as lco
import scipy as sp

def misses(x, y, mu, sigma):
    miss = 0
    S = 1000
    for i in range(x.size(dim=0)):
        sample = torch.normal(mean = float(mu[i]), std = float(sigma[i]), size = (1,S)).squeeze(dim=0)
        sample = torch.sort(sample)[0]
        color = [(0.3+0.1*abs(item-mu[i]), 0.4, 0.8-0.08*abs(item-mu[i])) for item in sample.detach().numpy()]
        plt.scatter(torch.ones(1000)*x[i], sample, c=color)
        plt.plot([-6,-4], torch.ones(2)*sample[int(np.floor(S*0.025))], 'k--')
        plt.plot([-6,-4], torch.ones(2)*sample[int(np.floor(S*0.975))], 'k--')
        plt.plot([-6,-4], torch.ones(2)*mu[i], 'b--')
        plt.scatter(x[i], y[i], edgecolors='k', c='g')   
        plt.show()
        if y[i] < sample[int(np.floor(S*0.025))] or y[i] > sample[int(np.ceil(S*0.975))-1]:
            miss += 1
    c1 = sp.stats.binom.ppf(0.025,y.size(dim=0),0.05)
    c3 = sp.stats.binom.ppf(0.5,y.size(dim=0),0.05)
    c2 = sp.stats.binom.ppf(0.975,y.size(dim=0),0.05)
    print(c1,c3,c2)
    print(f"CI: [{c1}, {c2}], Misses: {miss}, Within CI: {c1<=miss<=c2}")
    return c1, c2, miss

x = torch.linspace(-5,5,100)
y = np.load('../Finalized/test_data/data_100_y_0.npy')
opt = lco.Optimizer(x, y, order=2)
mu, sigma, conv_epoch, conv_time = opt.optimize(epochs = 3000, test = True)
l, u, miss = misses(x,torch.tensor(y),mu,sigma)