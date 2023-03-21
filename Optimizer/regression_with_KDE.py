import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt


'''
plt.scatter(x,y,zorder=2)
print(1-measure.total_mass())
measure.visualize()
plt.show()


plt.hist(measure.sample(10000),bins=50, density=True, range=[-4,4])
plt.hist(torch.randn(10000),bins=50, density=True, range=[-4,4], alpha=0.5)
plt.legend(['Model','True data'])
plt.show()
'''
N=13
w = torch.tensor([1/N]*N)
l = torch.linspace(-1,2,N)
w = torch.nn.parameter.Parameter(w)
l = torch.nn.parameter.Parameter(l)

m=pm.Measure(l,w)

sample_length = 20
h=1.06*N**(-1/5)
print(h)
x = torch.linspace(-1, 2, sample_length)
y = x + 1.1 + 0.1*torch.randn(sample_length)

def K(x):
    return 1/(np.sqrt(2*np.pi))*torch.exp(-x**2/2)

def loss_function(m:pm.Measure):
    alpha = m[0].locations
    weights = m[0].weights
    matrix = alpha.repeat(sample_length,1).transpose(0,1)-y + x
    return -(K(matrix/h).transpose(0,1)*(weights/(sample_length*h))).sum(dim=1).log().sum()

lr = 0.01
opt = pm.Optimizer(m, lr=lr)
opt.minimize(loss_function, max_epochs=1000, verbose=True, adaptive=False)

x=x.detach().numpy()
plt.scatter(x,y)
plt.xlim(-2, 4)
plt.ylim(0, 5)
m.visualize()
plt.show()
