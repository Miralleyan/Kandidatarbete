import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

N=200
data=torch.randn(1000)
data=torch.normal(0,3,size=(1,100))


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
'''
m=measure = pm.Measure(l, w)

sam=torch.linspace(-10,10,N)
h=1.06*len(sam)**(-1/5)
print(h)

def K(x):
    return torch.tensor(1/(np.sqrt(2*np.pi*h))*np.exp((-x**2/2).tolist()),requires_grad=True)


def KDE(x):
    return 1/(len(sam))*torch.matmul(K((x.reshape(-1,1)-sam)/h),m.weights.reshape(-1,1).double())


print("hello")
z=torch.linspace(-4,4,N)
t=KDE(z).detach().numpy()

plt.scatter(z.detach().numpy(),t)
plt.show()
#m.visualize()



'''

#index = torch.argmin(abs(l-data.view(-1,1)), dim=1)
def loss_fn(w):

    def K(x):
        return torch.tensor(1/(np.sqrt(2*np.pi*h))*np.exp((-x**2/2).tolist()),requires_grad=True)


    #sam=w[0].locations
    sam=torch.linspace(-10,10,N)
    h=1.06*len(sam)**(-1/5)

    return 1/(len(sam))*torch.matmul(K((x.reshape(-1,1)-sam)/h),w[0].weights.reshape(-1,1).double()).sum().log()
    
    #test=torch.tensor([1,2,3])
    #test1=torch.tensor([2,3,4])
    #test2=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    #tensor1 = torch.randn(10, 3)
    #print(tensor1)
    #print(test)
    #print(test2)
    #print(test1.reshape(-1,1))
    #print(test1.reshape(-1,1)-test)
    #print((test2-test))
    #print(torch.matmul(test2-test,test))
    #print([(K((xi.reshape(-1,1)-sam)/h)*w[0].weights).sum() for xi in data])
    #print(w[0].weights.reshape(-1,1))
    #print(x-sam)
    #print((x.reshape(-1,1)-sam)/h)
    #print(K((x.reshape(-1,1)-sam)/h))
    #print(K((x.reshape(-1,1)-sam)/h).double())
    #print(torch.matmul(K((x.reshape(-1,1)-sam)/h),w[0].weights.reshape(-1,1).double()))
    #print(x)
    #print((x.reshape(-1,1)-sam))
    #print(w[0].weights.reshape(-1,1).size())
    #Kx= 1/(len(sam))*torch.matmul(K((x.reshape(-1,1)-sam)/h),w[0].weights.reshape(-1,1).double())
    #print(Kx) 
    #print(w[0].weights)
    #print(Kx*w[0].weights)


    #sam=w[0].locations
    #x=w[0].locations
    #y=KDE(x).detach().numpy(
    #x=x.detach().numpy()
    #plt.scatter(x,y/sum(y))
    #plt.show()
    print("hello")
    z=torch.linspace(-4,4,N)
    t=KDE(z).detach().numpy()
    plt.scatter(z.detach().numpy(),t)
    plt.show
    plt.draw




    #return -KDE().log().sum()
    #return -w[0].weights[index].log().sum()


lr=0.01
measure = pm.Measure(l, w)
#print(loss_fn([measure]))
opt=pm.Optimizer([measure],lr=lr)


opt.minimize(loss_fn,smallest_lr=1e-10,verbose=False)
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