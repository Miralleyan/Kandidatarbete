import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

N=100
data=torch.randn(10000)
#data=torch.from_numpy(np.random.beta(1,2,1000))



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
    return -w[0].weights[index].log().sum()
    #return sum((y-w)**2)/len(w)
    #return -sum([torch.log(w[torch.nonzero(l==data[i].item()).item()]) for i in range(len(data))])


lr=0.0001
measure = pm.Measure(l, w)
opt=pm.Optimizer([measure],lr=lr)

'''
for epoch in range(5000):
    measure.zero_gradient()
    loss=loss_fn(measure.weights)
    loss.backward()
    opt.step(lr)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:<10} Loss: {loss:<10.0f} LR: {lr}')
'''

opt.minimize(loss_fn,smallest_lr=1e-10)



plt.scatter(x,y,zorder=2)
print(1-measure.total_mass())
measure.visualize()

'''
plt.hist(measure.sample(10000),bins=50, density=True, range=[-4,4])
plt.hist(torch.randn(10000),bins=50, density=True, range=[-4,4], alpha=0.5)
plt.legend(['Model','True data'])
plt.show()
'''
