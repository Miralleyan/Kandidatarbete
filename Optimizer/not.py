import torch
import numpy as np
import matplotlib.pyplot as plt

def h_1(x):
    return x*0+1
def h_2(x):
    return x
def h_3(x):
    return x**2

h = [h_1, h_2, h_3]

#def m(x, beta):
#    return sum([b[0] * h[i](x) for i, b in enumerate(beta)])
def m(mu, h_x):
    return (mu * h_x).sum()

#def sigma_2(x, sigma):
#    return sum([(b[1] * h[i](x))**2 for i, b in enumerate(beta)])
def sigma_2(sigma, h_x):
    return (sigma * h_x).pow(2).sum()

def log_normal_pdf(y, mu, sigma):
    return -1/2*(torch.log(2*np.pi*(sigma**2)) + ((y-mu)/sigma)**2)

def log_lik(x, y, beta, h_all):
    return sum([log_normal_pdf(y_i, m(beta[0], h_all[i]), sigma_2(beta[1], h_all[i])) for i, (y_i, x_i) in enumerate(zip(y,x))])

torch.seed = (1)
N = 200
x = torch.linspace(-4, 4, N)
y = torch.squeeze((torch.normal(mean=1.0,std=4,size=(1,N))) * x**2 + (-3+torch.randn(N)) * x + (torch.normal(mean=1.0,std=5,size=(1,N))), 0)
y.requires_grad = False

h_1_data = h_1(x)
h_2_data = h_2(x)
h_3_data = h_3(x)
h_all = torch.transpose(torch.stack([h_1_data, h_2_data, h_3_data], 0), 0, 1)

mu = torch.tensor([0., 0., 0.], dtype=float, requires_grad=True)
sigma = torch.tensor([1., 1., 1.], dtype=float, requires_grad=True)
beta = [mu, sigma]

plt.scatter(x, y)
plt.show()

optimizer = torch.optim.Adam(beta,lr=0.01, maximize=True)
for epoch in range(1000):
    optimizer.zero_grad()
    loss = log_lik(x, y, beta, h_all)
    loss.backward()
    optimizer.step()
    if epoch%10==0:
        print(epoch, mu, sigma)
mu = beta[0].detach().numpy()
sigma = beta[1].detach().numpy()
plt.show()
plt.scatter(x,y)
plt.plot(x, mu[2]*x**2+mu[1]*x+mu[0], 'r-')
plt.plot(x, mu[2]*x**2+mu[1]*x+mu[0]-2*sigma[0], 'r--')
plt.plot(x, mu[2]*x**2+mu[1]*x+mu[0]+2*sigma[0], 'r--')
plt.show()
print(beta)