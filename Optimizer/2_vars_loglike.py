import torch
import pytorch_measure as pm
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(15)
N = 20
x = torch.linspace(-1, 2, N)
y = 2 *torch.randn(N)*x + 1 + 0.2*torch.randn(N)

#plt.scatter(x, y)
#plt.show()

M = 20 # <- number of locations on measure

a = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)
b = pm.Measure(torch.linspace(0, 3, M), torch.ones(M) / M)


# return list[(sample (tensor -- list of locations), tensor -- probability)]
# assumes that the variables are independent
def unif_samples(ms: list[pm.Measure], n_samples):
    idx = (torch.rand((n_samples, len(ms))) * torch.tensor([len(m.locations) for m in ms])).long()
    locs = torch.cat([ms[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1)
    probs = torch.cat([ms[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1).prod(1)
    return (locs, probs)

def sample_meas(ms: list[pm.Measure], n_samples):
    idx = torch.cat([ms[i].sample(n_samples).unsqueeze(1) for i in range(len(ms))], 1).long()
    locs = torch.cat([ms[i].locations[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1)
    probs = torch.cat([ms[i].weights[idx[:, i]].unsqueeze(1) for i in range(len(ms))], 1).prod(1)
    return (locs, probs)

def loss_fn(measures: list[pm.Measure], n_samples=1000):
    return 


opt = pm.Optimizer([a, b], lr = 0.0002)
opt.minimize(loss_fn, max_epochs=10000, verbose = True)

a.visualize()
b.visualize()
aMax = a.locations[torch.argmax(a.weights)]
bMax = b.locations[torch.argmax(b.weights)]
plt.scatter(x,y)
plt.plot([-1,2], [-aMax+bMax, 2*aMax+bMax])
plt.show()
print(a.weights.grad)
print(b.weights.grad)