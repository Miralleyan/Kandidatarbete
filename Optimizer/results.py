import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_measure as pm

#-- Quadratic regression --

#- Theoretical solution -
def h_1(x):
    return x*0+1
def h_2(x):
    return x
def h_3(x):
    return x**2

h = [h_1, h_2, h_3]

def m(mu, h_x):
    return (mu * h_x).sum()

def sigma_2(sigma, h_x):
    return (sigma * h_x).pow(2).sum()

def log_normal_pdf(y, mu, sigma):
    return -1/2*(torch.log(2*np.pi*(sigma**2)) + ((y-mu)/sigma)**2)

def log_lik(x, y, beta, h_all):
    return sum([log_normal_pdf(y_i, m(beta[0], h_all[i]), sigma_2(beta[1], h_all[i])) for i, (y_i, x_i) in enumerate(zip(y,x))])

torch.seed = (1)
N = 500
x = torch.linspace(-4, 4, N)
y = torch.squeeze((torch.normal(mean=1.0,std=1,size=(1,N))) * x**2 + (-3+torch.randn(N)) * x + (torch.normal(mean=1.0,std=1,size=(1,N))), 0)
y.requires_grad = False

h_1_data = h_1(x)
h_2_data = h_2(x)
h_3_data = h_3(x)
h_all = torch.transpose(torch.stack([h_1_data, h_2_data, h_3_data], 0), 0, 1)

plt.scatter(x, y)
plt.show()

def runTheoretical(epochs):
    mu = torch.tensor([0., 0., 0.], dtype=float, requires_grad=True)
    sigma = torch.tensor([1., 1., 1.], dtype=float, requires_grad=True)
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
mu, sigma = runTheoretical(400)
sigma2 = 2*sigma
# plt.plot(x, mu[2]*x**2+mu[1]*x+mu[0], 'r-')
# plt.plot(x, (mu[2]+2*sigma[2])*x**2+(mu[1]+2*sigma[1])*x+mu[0]+2*sigma[0], 'b--')
# plt.plot(x, (mu[2]-2*sigma[2])*x**2+(mu[1]-2*sigma[1])*x+mu[0]-2*sigma[0], 'b--')
# plt.fill_between(x, (mu[2]+2*sigma[2])*x**2+(mu[1]+2*sigma[1])*x+mu[0]+2*sigma[0], (mu[2]-2*sigma[2])*x**2+(mu[1]-2*sigma[1])*x+mu[0]-2*sigma[0], alpha=0.2)
plt.plot(x, mu, 'r-')
plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--')
plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--')
plt.fill_between(x, [mu[i]+sigma[i] for i in range(N)], [mu[i]-sigma[i] for i in range(N)], alpha = 0.2)
ax = plt.gca()
ax.set_ylim([-10, 30])
plt.show()
print(mu, sigma)
# print(aMax,bMax,cMax)

# Linear regression two variables

#- Theoretical solution -
h = [h_1, h_2]

torch.seed = (1)
N = 500
x = torch.linspace(-4, 4, N)
y = torch.squeeze((-3+torch.randn(N)) * x + (torch.normal(mean=1.0,std=1,size=(1,N))), 0)
y.requires_grad = False

h_1_data = h_1(x)
h_2_data = h_2(x)
h_all = torch.transpose(torch.stack([h_1_data, h_2_data], 0), 0, 1)

plt.scatter(x, y)
plt.show()

def runTheoretical2(epochs):
    mu = torch.tensor([0., 0.], dtype=float, requires_grad=True)
    sigma = torch.tensor([1., 1.], dtype=float, requires_grad=True)
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

plt.scatter(x,y)
mu, sigma = runTheoretical2(400)
sigma2 = 2*sigma
plt.plot(x, mu, 'r-')
plt.plot(x, [mu[i]+sigma2[i] for i in range(N)], 'r--')
plt.plot(x, [mu[i]-sigma2[i] for i in range(N)], 'r--')
plt.fill_between(x, [mu[i]+sigma[i] for i in range(N)], [mu[i]-sigma[i] for i in range(N)], alpha = 0.2)
ax = plt.gca()
ax.set_ylim([-10, 30])

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

plt.show()