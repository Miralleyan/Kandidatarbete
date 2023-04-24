import pytorch_measure as pm
import torch


n=100
s=3
mu=7
data=torch.randn(n)*s+mu

print(data)

#If both kernel and data are unimodal normal distribution this is a good alternative
sigma=torch.std(data)
A=min(sigma,(torch.quantile(data,0.75)-torch.quantile(data,0.25))/1.35)
h=0.9*A*n**(-1/5)






####
from KDEpy import TreeKDE
import numpy as np
import matplotlib.pyplot as plt
import pytorch_measure as pm

N = 17 # number of atoms
M = 2000 # Number of datapoints
amin = -5
amax = 3
verbose = True
dev = 'cpu'


torch.manual_seed(1)

def regression_model(a, x):
    return a+x

x = torch.linspace(0, 10, M).view(-1, 1)
data = regression_model(torch.randn(M).to(dev) - 2, x.view(1, -1)).view(-1, 1)
w = torch.rand(N,dtype=torch.float).to(dev)
w = torch.nn.parameter.Parameter(w/w.sum())
l = torch.linspace(amin, amax, N, requires_grad=False).to(dev)





def KDENLLLoss(m):
    d=m[0].sample(M).numpy()
    y=TreeKDE(kernel='gaussian', bw='ISJ').fit(d).evaluate(data.numpy())
    return torch.from_numpy(y).requires_grad_().log().sum()


measure = pm.Measure(locations=l, weights=w, device=dev)
opt_KDE = pm.Optimizer([measure], lr=1e-1)
new_mesKDE = opt_KDE.minimize(KDENLLLoss, verbose=True, print_freq=10, max_epochs=1000, tol_const=1e-3, adaptive=True)




####
'''
from KDEpy import TreeKDE
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate a distribution and some multimodal data
data = norm(loc=0, scale=1)
#print(data.rvs(2**13))


# Compute density estimates using 'silverman'
x, y = TreeKDE(kernel='gaussian', bw='silverman').fit(data.rvs(2**10)).evaluate()
print(x,y)
plt.plot(x, y, label='KDE /w silverman')

# Compute density estimates using 'ISJ' - Improved Sheather Jones
y = TreeKDE(kernel='gaussian', bw='ISJ').fit(data.rvs(2**10)).evaluate(x)
plt.plot(x, y, label='KDE /w ISJ')

plt.plot(x, data.pdf(x), label='True pdf')
plt.grid(True, ls='--', zorder=-15); plt.legend();
plt.show()
'''