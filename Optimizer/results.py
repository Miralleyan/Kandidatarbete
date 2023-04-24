import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_measure as pm

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
    for i in range(x.size(dim=0)):
        sample = torch.normal(mean = float(mu[i]), std = float(sigma[i]), size = (1,1000)).squeeze(dim=0)
        sample = torch.sort(sample)[0]
        if y[i] < sample[25] or y[i] > sample[974]:
            miss += 1
    return miss

# Linear regression two variables
h = [h_1, h_2]

torch.seed = (2)
N = 500
x = torch.linspace(-4, 4, N)
y = torch.squeeze((-3+torch.randn(N)) * x + (torch.normal(mean=1.0,std=1,size=(1,N))), 0)
y.requires_grad = False

#- Theoretical solution -
h_1_data = h_1(x)
h_2_data = h_2(x)
h_all = torch.transpose(torch.stack([h_1_data, h_2_data], 0), 0, 1)

plt.scatter(x, y)
plt.show()

mu = torch.tensor([0., 0.], dtype=float, requires_grad=True)
sigma = torch.tensor([1., 1.], dtype=float, requires_grad=True)
def runTheoretical(epochs):
    beta = [mu, sigma]
    optimizer = torch.optim.Adam(beta,lr=0.1, maximize=True)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = log_lik(x, y, beta, h_all)
        loss.backward()
        optimizer.step()
        if epoch%10==0:
            print(epoch, mu, sigma)
    # mu = beta[0].detach().numpy()
    # sigma = beta[1].detach().numpy()
    return [m(mu, h_all[i,:]).detach().numpy() for i in range(N)], [(sigma_2(sigma, h_all[i,:])**0.5).detach().numpy() for i in range(N)]

plt.scatter(x,y,alpha=0.5)
mu, sigma = runTheoretical(400)
sigma2 = 2*sigma
plt.plot(x, mu, 'r-')
plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--') # Upper bound confidence interval
plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--') # Lower bound confidence interval
plt.fill_between(x, [mu[i]+sigma[i] for i in range(N)], [mu[i]-sigma[i] for i in range(N)], alpha = 0.2)
ax = plt.gca()
ax.set_ylim([-10, 30])

#- Our method -
def linModel(x,params):
    return params[1]*x+params[0]

M = 50
a = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
opt = pm.Optimizer([b,a], 'KDEnll', lr = 0.1)
[b,a] = opt.minimize([x,y], linModel, max_epochs=400)
aMean = torch.sum(a.locations*a.weights).detach()
bMean = torch.sum(b.locations*b.weights).detach()
plt.plot(x, aMean*x+bMean, 'b-')
print(misses(x, y, mu, sigma))

plt.show()

#-- Quadratic regression --
torch.seed = (1)
N = 500
x = torch.linspace(-4, 4, N)
y = torch.squeeze((torch.normal(mean=1.0,std=1,size=(1,N))) * x**2 + (-3+torch.randn(N)) * x + (torch.normal(mean=1.0,std=1,size=(1,N))), 0)
y.requires_grad = False

#- Theoretical solution -
h = [h_1, h_2, h_3]

h_1_data = h_1(x)
h_2_data = h_2(x)
h_3_data = h_3(x)
h_all = torch.transpose(torch.stack([h_1_data, h_2_data, h_3_data], 0), 0, 1)

plt.scatter(x, y)
plt.show()

#- Our method -
def model(x,params):
    return params[2]*x**2+params[1]*x+params[0]

M = 50
a = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
c = pm.Measure(torch.linspace(-10, 10, M), torch.ones(M) / M)
opt = pm.Optimizer([c,b,a], 'KDEnll', lr = 0.1)
[c,b,a] = opt.minimize([x,y], model, max_epochs=400)
aMean = torch.sum(a.locations*a.weights).detach()
bMean = torch.sum(b.locations*b.weights).detach()
cMean = torch.sum(c.locations*c.weights).detach()

plt.scatter(x,y, alpha=0.5)
plt.plot(x, aMean*x**2+bMean*x+cMean, 'b-')
mu = torch.tensor([0., 0.], dtype=float, requires_grad=True)
sigma = torch.tensor([1., 1.], dtype=float, requires_grad=True)
mu, sigma = runTheoretical(400)
sigma2 = 2*sigma
plt.plot(x, mu, 'r-')
plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--') # Upper bound confidence interval
plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--') # Lower bound confidence interval
plt.fill_between(x, [mu[i]+sigma[i] for i in range(N)], [mu[i]-sigma[i] for i in range(N)], alpha = 0.2)
ax = plt.gca()
ax.set_ylim([-10, 30])
plt.show()
print(misses(x,y,mu,sigma))


#- Constant -
torch.seed = (1)
N = 500
x = torch.linspace(-4, 4, N)
y = torch.squeeze((torch.normal(mean=1.0,std=1,size=(1,N))) * x**2 + (-3+torch.randn(N)) * x + (torch.normal(mean=1.0,std=1,size=(1,N))), 0)
y.requires_grad = False

#- Theoretical solution -
h = h_1

h_1_data = h_1(x)
h_all = torch.transpose(torch.stack([h_1_data, h_2_data, h_3_data], 0), 0, 1)

plt.scatter(x, y)
plt.show()

mu = torch.tensor([0.], dtype=float, requires_grad=True)
sigma = torch.tensor([1.], dtype=float, requires_grad=True)
mu, sigma = runTheoretical(400)
sigma2 = 2*sigma
plt.plot(x, mu, 'r-')
plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--') # Upper bound confidence interval
plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--') # Lower bound confidence interval
plt.show()