import matplotlib.pyplot as plt
import torch
import PyTorchMeasure as pm
import numpy as np

######## Setup #########
N = 31  # number of atoms
M = 2000  # Number of datapoints
data_mean = -2.
data_std = 1.
amin = -6.
amax = 2.
h = 0.4         #1.06*N**(-1/5)
verbose = True
dev = 'cpu'


torch.manual_seed(1)
#####################
def regression_model(a, x):
    return a+x

x = torch.linspace(0, 10, M).view(-1, 1)
data = regression_model(
        torch.tensor(np.random.normal(data_mean, data_std, M), dtype=torch.float),
        x.view(1, -1)
    ).view(-1, 1)
w = torch.rand(N, dtype=torch.float).to(dev)
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

# K( (y - yj) / h )
kde_mat = K((data.view(-1,1) - regression_model(l, x)) / h)

def KDENLLLoss(m):
    return -(torch.matmul(kde_mat, m[0].weights.view(-1,1)) / (M*h)).log().sum()

measure = pm.Measure(locations=l, weights=w, device=dev)

opt_NLL = pm.Optimizer([measure], lr=1e-1)
new_mes = opt_NLL.minimize(NLLLoss, verbose=verbose, print_freq=100, max_epochs=1000, tol_const=1e-2, adaptive=True)

opt_KDE = pm.Optimizer([measure], lr=1e-1)
new_mesKDE = opt_KDE.minimize(KDENLLLoss, verbose=verbose, print_freq=100, max_epochs=1000, tol_const=1e-2, adaptive=True)

 #Create true values
xs = torch.linspace(amin, amax, 500)
y=1/(np.sqrt(2*np.pi)*data_std)*torch.exp(-(xs-data_mean)**2/(2*data_std**2))

y_mes = torch.matmul(K((xs.view(-1, 1) - l)/h), new_mes[0].weights.view(-1, 1)) / (h)
y_mesKDE = torch.matmul(K((xs.view(-1, 1) - l)/h), new_mesKDE[0].weights.view(-1, 1)) / (h)
#plt.bar(l - 0.05, new_mes[0].weights.tolist(), width = 0.1, label='NLLLoss')
#plt.bar(l + 0.05, new_mesKDE[0].weights.tolist(), width = 0.1, label='KDELoss')
plt.plot(xs.detach(), y_mesKDE.detach(), label='KDELoss')
plt.plot(xs.detach(), y_mes.detach(), label='NLLLoss')
plt.plot(xs.detach(), y, zorder=2, label="True distribution")
plt.legend()

plt.show()