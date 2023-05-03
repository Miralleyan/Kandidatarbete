import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import time
import linear_combination_optimizer as lco
import json

# Polynomial nn method
def eval_powers_of_x(x, n):
    return x.pow(torch.arange(n))
def eval_even_powers_of_x(x, n):
    return x.pow(2 * torch.arange(n))

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

class NormalPolynomialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mean = None
        self.std = None
        
        self.N_mean = 2 # constant, linear, quadratic, ... terms
        self.N_var = 2

        self.poly_multipliers_mean = torch.nn.parameter.Parameter(torch.rand(self.N_mean))
        self.poly_multipliers_var = torch.nn.parameter.Parameter(torch.rand(self.N_var))
        self.var_shift = torch.nn.parameter.Parameter(torch.tensor(0.1))

        self.mean_layer = torch.nn.Linear(self.N_mean, 1)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(self.N_var, 1),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        self.mean = self.mean_layer(self.poly_multipliers_mean * eval_powers_of_x(x, self.N_mean))
        self.var = self.var_layer(self.poly_multipliers_var * eval_even_powers_of_x(x, self.N_var)) + self.var_shift ** 2
        return self.mean, self.var
    
def log_k_with_var(mean, var, y):
    return -0.5*torch.log(var) - (y - mean)**2 / (2 * var)


for length in [100, 500, 1000]:
    success=[]
    tid=[]
    end_epoch=[]
    means=[]
    std=[]
    for i in range(50):
        x = torch.linspace(-5, 5, length)
        y = np.load(f'../Finalized/test_data/data_{length}_y_lin_{i}.npy')
        y = torch.tensor(y)
        x_unsq = x.view(-1, 1)
        y = y.view(-1, 1)
        model = NormalPolynomialModel()
        opt = torch.optim.Adam(model.parameters())

        max_epoch = 300000
        conv_epoch = max_epoch
        conv_time = float('inf')
        old_loss = float('inf')
        t1 = time.time()
        for epoch in range(max_epoch):
            opt.zero_grad()
            mean, var = model(x_unsq)
            log_likelyhood = log_k_with_var(mean, var, y).sum()
            loss = -log_likelyhood
            loss.backward()
            opt.step()
            if torch.abs(old_loss-loss) < 1e-4:
                t2 = time.time()
                conv_time = t2-t1
                conv_epoch = epoch
                break
            old_loss = loss

            if epoch % 1000 == 0:
                print(f'{epoch}:: Loss = {loss.item()}')
        m, var = model(x_unsq)
        s = var.sqrt()
        m = m.detach().numpy()
        s = s.detach().numpy()

        l, u, miss = misses(x,torch.tensor(y),m,s)
        success.append(l<=miss and miss<=u)
        if conv_time != float('inf'):
            tid.append(conv_time)
        end_epoch.append(conv_epoch)
        means.append(m.tolist())
        std.append(s.tolist())
    
    results = [means, std, sum(tid)/len(tid), sum(end_epoch)/(len(end_epoch)), float(100*sum(success)/len(success))]
    with open(f"Poly_results/poly_results_{length}_y_lin.json", "w") as outfile:
        outfile.write(json.dumps(results))