import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

N=100
data=torch.randn(1000)
M=N


w = torch.tensor([1/N]*N)#Weights
l = torch.linspace(-4,4,N)
w = torch.nn.parameter.Parameter(w)
l = torch.nn.parameter.Parameter(l)


mu=0 #Create true values
sigma=1
x=torch.linspace(-4,4,N)
y=1/(np.sqrt(2*np.pi))*torch.exp(-(x-mu)**2/(2*sigma**2))
y/=sum(y) #Normalize

h=0.2




#index = torch.argmin(abs(l-data.view(-1,1)), dim=1)
def loss_fn(w):
    def K(x):
        return 1/(np.sqrt(2*np.pi))*torch.exp(-x**2/2)

    def KDE(x):
        return torch.tensor([1/(N*h) *(K((xi-w[0].locations).sum()/h)).item() for xi in x], requires_grad=True)

    return -KDE(data).log().sum()
    #return -w[0].weights[index].log().sum()
    #return sum((y-w)**2)/len(w)
    #return -sum([torch.log(w[torch.nonzero(l==data[i].item()).item()]) for i in range(len(data))])


'''
def error(x,y): # a is location in measure (scalar), for example slope in linear regression
    return ((x - y).pow(2)).sum()

def loss_fn(measures):
    errors = torch.tensor([error(data, y) for j in range(M)])
    return torch.dot(errors, measures[0].weights)
    '''


lr=0.1
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

opt.minimize(loss_fn,smallest_lr=1e-10,verbose=True, tol_const=1e-40)



plt.scatter(x,y,zorder=2)
print(1-measure.total_mass())
measure.visualize()

'''
plt.hist(measure.sample(10000),bins=50, density=True, range=[-4,4])
plt.hist(torch.randn(10000),bins=50, density=True, range=[-4,4], alpha=0.5)
plt.legend(['Model','True data'])
plt.show()
'''
