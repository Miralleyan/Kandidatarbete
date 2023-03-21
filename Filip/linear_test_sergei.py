import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_measure import TorchMeasure, MeasureMinimizer

N=41
data=torch.randn(2000)
#data=torch.from_numpy(np.random.beta(1,2,1000))
#torch.manual_seed(1)



w = torch.tensor([1/N]*N)#Weights
l = torch.linspace(-4,4,N)
w = torch.nn.parameter.Parameter(w)
l = torch.nn.parameter.Parameter(l)


mu=0 #Create true values
sigma=1
x=torch.linspace(-4,4,N)
y=1/(np.sqrt(2*np.pi)*sigma)*torch.exp(-(x-mu)**2/(2*sigma**2))
y/=sum(y) #Normalize

index = torch.argmin(abs(l-data.view(-1,1)), dim=1)
def loss_fn(w):
    return -w.weights[index].log().sum()



lr=0.1
# measure = pm.Measure(l, w)
# opt=pm.Optimizer(measure,lr=lr)

measure = TorchMeasure(l, w) # random starting measure

opt = MeasureMinimizer(measure, loss_fn, learning_rate=lr)
opt.minimize(print_each_step=1, silent=False, max_no_steps=1000, adaptive=False)

#opt.minimize(loss_fn,verbose=True)
# opt.minimize(loss_fn,verbose=True, max_epochs=300, adaptive=True, print_freq=100)


plt.scatter(x,y,zorder=2)
# print(1-measure.total_mass())
# measure.visualize()
# plt.show()

'''
plt.hist(measure.sample(10000),bins=50, density=True, range=[-4,4])
plt.hist(torch.randn(10000),bins=50, density=True, range=[-4,4], alpha=0.5)
plt.legend(['Model','True data'])
plt.show()
'''






#likmax.minimize(print_each_step=10, tol_const=0.01, adaptive=True)

# print(likmax.grad)
# likmax.mes.visualize()
amax=4
amin=-4
no_atoms = N
locations_grid_size = (amax - amin) / (no_atoms - 1)
bins=torch.linspace(amin - locations_grid_size / 2, amax + locations_grid_size / 2, no_atoms + 1)
h,b = torch.histogram(y - x, bins=bins)
midpoints = (b[1:] + b[:-1]) / 2

# side-by-side histogram. Can it be done smarter?
plt.bar(midpoints - 0.1, opt.mes.weights.tolist(), width = 0.2, label='fitted')
#plt.bar(midpoints + 0.1, h / h.sum(), width = 0.2, label='data')
plt.legend(loc='upper right')
plt.show()

