import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

N=20
data=torch.randn(1000)
#data=torch.normal(0,3,size=(1,100))


w = torch.tensor([1/N]*N)#Weights
l = torch.linspace(-4,4,N)
w = torch.nn.parameter.Parameter(w)
l = torch.nn.parameter.Parameter(l)


mu=0 #Create true values
sigma=1
x=torch.linspace(-4,4,N)
y=1/(np.sqrt(2*np.pi))*torch.exp(-(x-mu)**2/(2*sigma**2))
y/=sum(y) #Normalize
#w=torch.tensor(y)
#'''
m=measure = pm.Measure(l, w)

sam=torch.linspace(-10,10,N)
h=1.06*len(sam)**(-1/5)
print(h)

def K(x):
    return torch.tensor(1/(np.sqrt(2*np.pi*h))*np.exp((-x**2/2).tolist()),requires_grad=True)


def K(x):
    return 

def KDE(x):
    return 1/(len(sam))*torch.matmul(K((x.reshape(-1,1)-sam)/h),m.weights.reshape(-1,1).double())

'''
print("hello")
z=torch.linspace(-4,4,N*10)
t=KDE(z).detach().numpy()
t=t/sum(t)

plt.scatter(z.detach().numpy(),t)
m.visualize()
plt.show()
'''



#'''

#index = torch.argmin(abs(l-data.view(-1,1)), dim=1)
def loss_fn(w):
    #sam=w[0].locations
    sam=torch.linspace(-4,4,N)
    h=1.06*len(sam)**(-1/5)
    
    #sam=w[0].sample(N)

    def K(x):
        return torch.tensor(1/(np.sqrt(2*np.pi*h))*np.exp((-x**2/2).tolist()),requires_grad=True)

   
    #KDE=(100/N*torch.matmul(w[0].weights.double(),K((sam.reshape(-1,1)-x)/h)[?]))
    KDE=(1/(len(sam))*torch.matmul(K((x.reshape(-1,1)-sam)/h),w[0].weights.reshape(-1,1).double()))
    #KDE=(10/(len(sam))*K((x.reshape(-1,1)-sam)/h).sum(0))
    #t=KDE.detach().numpy()

    #w[0].visualize()
    #plt.scatter(sam.detach().numpy(),t)
    
    #plt.show()
    KDE=KDE/sum(KDE)
    return -KDE.log().sum()
    
    print("hello")
    z=torch.linspace(-4,4,N)
    t=KDE(z).detach().numpy()
    plt.scatter(z.detach().numpy(),t)
    plt.show
    plt.draw



lr=0.001
measure = pm.Measure(l, w)
opt=pm.Optimizer([measure],lr=lr)


opt.minimize(loss_fn,smallest_lr=1e-10,verbose=False, print_freq=100)
measure.visualize()
plt.show()
#'''

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