import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_measure as pm
import scipy as sp
import time

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

def log_lik(x, y, beta, h_all):
    return sum([log_normal_pdf(y_i, m(beta[0], h_all[i]), sigma_2(beta[1], h_all[i])) for i, (y_i, x_i) in enumerate(zip(y,x))])


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

def runTheoretical(x, y, h_all, mu, sigma, epochs):
    beta = [mu, sigma]
    optimizer = torch.optim.Adam(beta,lr=0.1, maximize=True)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = log_lik(x, y, beta, h_all)
        loss.backward()
        optimizer.step()
        if epoch%10==0:
            print(epoch, mu, sigma)
    return [m(mu, h_all[i,:]).detach().numpy() for i in range(x.size(dim=0))], [(sigma_2(sigma, h_all[i,:])**0.5).detach().numpy() for i in range(x.size(dim=0))]

# Linear regression two variables
# Linear model
def linModel(x,params):
    return params[1]*x+params[0]

def linTest(x, y):
    M = 30
    a = pm.Measure(torch.linspace(-5, 5, M), torch.ones(M) / M)
    b = pm.Measure(torch.linspace(-5, 5, M), torch.ones(M) / M)
    opt = pm.Optimizer([b,a], 'KDEnll', lr = 0.1)
    [b,a] = opt.minimize([x,y], linModel, max_epochs=1000)
    aMean = torch.sum(a.locations*a.weights).detach()
    bMean = torch.sum(b.locations*b.weights).detach()
    checker = pm.Check(opt, linModel, x, y, normal=False)
    checker.check()
    opt.visualize()

    plt.show()

    h = [h_1, h_2]

    #- Theoretical solution -
    h_1_data = h_1(x)
    h_2_data = h_2(x)
    h_all = torch.transpose(torch.stack([h_1_data, h_2_data], 0), 0, 1)

    plt.scatter(x, y)
    plt.show()

    mu = torch.tensor([0., 0.], dtype=float, requires_grad=True)
    sigma = torch.tensor([1., 1.], dtype=float, requires_grad=True)

    plt.scatter(x,y,alpha=0.5)
    t1 = time.time()
    mu, sigma = runTheoretical(x,y,h_all,mu, sigma, 1000)
    t2 = time.time()
    print(f'Time: {t2-t1}')
    misses(x, y, mu, sigma)
    sigma2 = 2*sigma
    plt.plot(x, mu, 'r-')
    plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--') # Upper bound confidence interval
    plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--') # Lower bound confidence interval
    plt.fill_between(x, [mu[i]+sigma[i] for i in range(N)], [mu[i]-sigma[i] for i in range(N)], alpha = 0.2)
    plt.plot(x, aMean*x+bMean, 'b-')
    plt.show()
    # ax = plt.gca()
    # ax.set_ylim([-10, 30])

#-- Quadratic regression --

#- Our method -
def quadModel(x,params):
    return params[2]*x**2+params[1]*x+params[0]

def quadTest(x, y):
    #- Theoretical solution -
    h = [h_1, h_2, h_3]

    h_1_data = h_1(x)
    h_2_data = h_2(x)
    h_3_data = h_3(x)
    h_all = torch.transpose(torch.stack([h_1_data, h_2_data, h_3_data], 0), 0, 1)

    plt.scatter(x, y)
    plt.show()

    M = 20
    a = pm.Measure(torch.linspace(-3, 3, M), torch.ones(M) / M)
    b = pm.Measure(torch.linspace(-3, 3, M), torch.ones(M) / M)
    c = pm.Measure(torch.linspace(-3, 3, M), torch.ones(M) / M)
    opt = pm.Optimizer([c,b,a], 'KDEnll', lr = 0.1)
    [c,b,a] = opt.minimize([x,y], quadModel, max_epochs=1000)
    aMean = torch.sum(a.locations*a.weights).detach()
    bMean = torch.sum(b.locations*b.weights).detach()
    cMean = torch.sum(c.locations*c.weights).detach()
    checker = pm.Check(opt, quadModel, x, y)
    checker.check()

    plt.scatter(x,y, alpha=0.5)
    plt.plot(x, aMean*x**2+bMean*x+cMean, 'b-')
    mu = torch.tensor([0., 0., 0.], dtype=float, requires_grad=True)
    sigma = torch.tensor([1., 1., 1.], dtype=float, requires_grad=True)
    t1 = time.time()
    mu, sigma = runTheoretical(x, y, h_all, mu, sigma, 1000)
    t2 = time.time()
    print(f'Time: {t2-t1}') # Our code takes 14.49s, while Sergei's takes 51.12s
    sigma2 = 2*sigma
    plt.plot(x, mu, 'r-')
    plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--') # Upper bound confidence interval
    plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--') # Lower bound confidence interval
    plt.fill_between(x, [mu[i]+sigma[i] for i in range(N)], [mu[i]-sigma[i] for i in range(N)], alpha = 0.2)
    ax = plt.gca()
    ax.set_ylim([-10, 30])
    plt.show()
    misses(x,y,mu,sigma)

def constModel(x,params):
    return params[0]*1
#- Constant -
def normTest(x, y):

    #- Theoretical solution -
    h = h_1

    h_1_data = h_1(x)
    h_all = torch.transpose(torch.stack([h_1_data], 0), 0, 1)

    plt.scatter(x, y)
    plt.show()

    mu = torch.tensor([0.], dtype=float, requires_grad=True)
    sigma = torch.tensor([1.], dtype=float, requires_grad=True)
    mu, sigma = runTheoretical(x, y, h_all, mu, sigma, 1000)
    sigma2 = 2*sigma
    misses(x,y,mu,sigma)

    # plt.scatter(x,y, alpha=0.5)
    # plt.plot(x, mu, 'r-')
    # plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--') # Upper bound confidence interval
    # plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--') # Lower bound confidence interval
    # plt.show()

    #- Our method -
    M = 50
    a = pm.Measure(torch.linspace(-4, 4, M), torch.ones(M) / M)
    opt = pm.Optimizer([a], 'KDEnll', lr = 0.1)
    [a] = opt.minimize([x,y], constModel, max_epochs=1000)
    checker = pm.Check(opt, constModel, x, y, 0.05)
    checker.check()

    counts, bins = np.histogram(y,bins=20)
    plt.stairs(counts/sum(counts), bins)
    plt.plot(x, sp.stats.norm.pdf(x, mu, sigma))
    plt.plot(a.locations.detach().numpy(), a.weights.detach().numpy())
    plt.show()


# N = 1000
# x = torch.linspace(-10, 10, N)
# y = torch.squeeze((torch.normal(mean=1.0,std=1,size=(1,N))), 0)
# normTest(x, y)

# N = 100
# x = torch.linspace(-5, 5, N)
# y = torch.squeeze((-2+torch.randn(N)) * x + (torch.normal(mean=2.0,std=1,size=(1,N))), 0)
# linTest(x, y)

N = 1000
x = torch.linspace(-10, 10, N)
y = torch.squeeze((torch.normal(mean=1.0,std=1,size=(1,N))) * x**2 + (-3+torch.randn(N)) * x + (torch.normal(mean=1.0,std=1,size=(1,N))), 0)
quadTest(x, y)