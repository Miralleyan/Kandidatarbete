# File for comparing results of one measure linear regression
# between methods

import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time

#torch.manual_seed(30) # <-- if seed is wanted
N = 1000
x = torch.linspace(-3, 5, N)


# Plot the data points
#plt.scatter(x, y)
plt.show()

# Number of locations of measure
M = 30

# Linear regression model
def regression_model(x,list):
     return x+list[0]

runs = 20
alpha = torch.randn(runs)

success=[]
means_dist = [[],[]]
std_devs_dist = [[],[]]
for i in range(runs):
    # Measure for slope (a) and intercept (b) of linear model
    a = pm.Measure(torch.linspace(-4, 4, M), torch.ones(M) / M)


    measures = [a]
    y = (torch.randn(N)+alpha[i])+x
    # Instance of optimizer
    opt = pm.Optimizer(measures, "KDEnll", lr = 0.1)
    # Call to minimizer
    new_mes=opt.minimize([x,y],regression_model,max_epochs=1000,verbose = True, print_freq=100, smallest_lr=1e-10)
    # Visualize measures and gradient
    new_mes[0].visualize()
    #plt.show()

    check=pm.Check(opt,regression_model,x,y,normal=True,Return=True)
    l,u,miss=check.check()
    #check.check()
    success.append(l<=miss and miss<=u)

    a_mean = sum(a.weights*a.locations)
    means_dist[0].append(torch.abs(alpha[i]-a_mean))
    std_devs_dist[0].append(torch.abs(1-torch.sqrt(sum(a.weights*(a.locations-a_mean)**2))))


print(f'{sum(success)} successes')
print(f'Our method succeeds {100*sum(success)/runs}% of the time')
print(f'Average distance from correct mean: {sum(means_dist[0])/runs}')
print(f'Average distance from correct variance: {sum(std_devs_dist[0])/runs}')


# Linear combination method
# Functions for linear combination
def h_1(x):
    return x*0+1
def h_2(x):
    return x
def h_3(x):
    return x**2

def m(mu, h_x):
    return (mu * h_x).sum()

def sigma_2(sigma, h_x):
    return (sigma * h_x).pow(2).sum()

def log_normal_pdf(y, mu, sigma_2):
    return -1/2*(torch.log(2*np.pi*(sigma_2)) + ((y-mu)**2/sigma_2))

def log_lik(y, beta, h_all):
    return sum([log_normal_pdf(y_i, m(beta[0], h_all[i]), sigma_2(beta[1], h_all[i])) for i, y_i in enumerate(y)])


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

# Linear combs
def runTheoretical(x, y, h_all, mu, sigma, epochs):
    beta = [mu, sigma]
    optimizer = torch.optim.Adam(beta,lr=0.1, maximize=True)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = log_lik(y, beta, h_all)
        loss.backward()
        optimizer.step()
        if epoch%10==0:
            print(epoch, mu, sigma)
    return [m(mu, h_all[i,:]).detach().numpy() for i in range(x.size(dim=0))], [(sigma_2(sigma, h_all[i,:])**0.5).detach().numpy() for i in range(x.size(dim=0))]

success=[]
for i in range(runs):
    y = (torch.randn(N)+alpha[i])+x
    h = [h_1]

    #- Theoretical solution -
    h_1_data = h_1(x)
    h_all = torch.transpose(torch.stack([h_1_data], 0), 0, 1)

    mu = torch.tensor([0., 0.], dtype=float, requires_grad=True)
    sigma = torch.tensor([1., 1.], dtype=float, requires_grad=True)

    t1 = time.time()
    mu, sigma = runTheoretical(x,y,h_all,mu,sigma,1000)
    t2 = time.time()
    print(f'Time: {t2-t1}')
    l, u, miss = misses(x,y,mu,sigma)
    success.append(l<=miss and miss<=u)

    # a_mean = sum(mu[0])/N
    # means_dist[0].append(torch.abs(alpha[i]-a_mean))
    # std_devs_dist[0].append(torch.abs(1-sum(sigma[0])/N))

print(f'{sum(success)} successes')
# print(f'Average distance from correct mean: {sum(means_dist[0])/runs}')
# print(f'Average distance from correct variance: {sum(std_devs_dist[0])/runs}')
print(f'Linear combination method succeeds {100*sum(success)/runs}% of the time')