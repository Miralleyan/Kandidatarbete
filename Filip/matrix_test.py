import torch



t1 = torch.tensor([1.,2.,3.])
w = torch.tensor([0.1,0.2,0.3])
t2 = torch.tensor([1.1,2.1,3.1,4.1])

m1 = t1.repeat(4,1).transpose(0,1) - t2
print(m1)

def f(x):
    return x**2

m2 = f(m1)
print(m2)

m3 = m2.transpose(0,1) + w
print(m3)
m4 = m3.sum(dim=1)
print(m4)