a=[12,3,4]
import pytorch_measure as pm
import torch
b=[1]
b.append(a)
print(b)


def model(list,x):
    a+b*x+cx**2


m=pm.Measure(torch.tensor([1,2,3]),torch.tensor([0.1,0.4,0.5]))
print(m.sample(100).size())