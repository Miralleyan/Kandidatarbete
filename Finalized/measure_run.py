import matplotlib.pyplot as plt
import torch
import PyTorchMeasure as pm
import numpy as np

N = 17 # number of atoms
M = 2000 # Number of datapoints
amin = -6
amax = 2
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

index = []
for i in range(M):
    ab = (regression_model(l, x[i]) - data[i]).abs()
    index.append(torch.argmin(ab))

def NLLLoss(m:list[pm.Measure]):
    return -(m[0].weights[index]).log().sum()

def K(d):
        return 1/np.sqrt(2*np.pi)*np.exp(-d**2/2)
h=1.06*M**(-1/5)
# K( (y - yj) / h )
kde_mat = K((data.view(-1,1) - regression_model(l, x)) / h)

def KDENLLLoss(m):
    return -(torch.matmul(kde_mat, m[0].weights.view(-1,1)) / (M*h)).log().sum()

measure = pm.Measure(locations=l, weights=w, device=dev)

opt_NLL = pm.Optimizer([measure], lr=1e-1)
new_mes = opt_NLL.minimize(NLLLoss, verbose=False, print_freq=100, max_epochs=1000, tol_const=1e-2, adaptive=True)

opt_KDE = pm.Optimizer([measure], lr=1e-1)
new_mesKDE = opt_KDE.minimize(KDENLLLoss, verbose=False, print_freq=100, max_epochs=1000, tol_const=1e-2, adaptive=True)

mu=0 #Create true values
sigma=1
xs = l.detach()
y=1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-(xs+2-mu)**2/(2*sigma**2))
y/=sum(y) #Normalize


plt.bar(l - 0.05, new_mes[0].weights.tolist(), width = 0.1, label='NLLLoss')
plt.bar(l + 0.05, new_mesKDE[0].weights.tolist(), width = 0.1, label='KDELoss')
plt.scatter(xs, y, zorder=2, label="True distribution")
plt.legend()

plt.show()