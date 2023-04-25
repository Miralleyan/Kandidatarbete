import torch
import matplotlib.pyplot as plt
import numpy as np
#%%
N = 500

def real_model(x):
    '''
    y = N(2,1)x + N(3,1)
    Real mean rm(x)=2x + 3
    Real st.dev. rs(x) = sqrt(x**2+1)
    '''
    y = (torch.randn(1, N) + 2) * x + 3 + torch.randn(1, N)  #
    return y

    # y = 0.2 * (3 + torch.randn(1, N)) + 0.5*(2 + torch.randn(1, N)) * x #
    # return y

x = torch.linspace(-3, 4, N).view(-1, 1)
y = real_model(x.squeeze()).view(-1, 1)
# y = real_model(x.squeeze()).view(-1, 1)

rm = 3+2*x
rs = np.sqrt(1+x**2)
q90 = 1.64485 # -scipy.stats.norm.ppf(0.1)
cil = (rm - q90 * rs).squeeze()
cir = (rm + q90 * rs).squeeze()

plt.fill_between(x.squeeze(), cil, cir, color = 'orange', alpha=0.5)
plt.scatter(x, y, marker='.', c='red')
plt.plot(x, rm, '-', c="green")
plt.show()

#%%
class NormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mean = None
        self.var = None
        self.shift = torch.nn.Parameter(torch.tensor([1.], requires_grad=True))

        self.mean_layer = torch.nn.Linear(1, 1)
        self.var_layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        self.mean = self.mean_layer(x) # linear mean
        self.var = self.var_layer(x) ** 2 + self.shift ** 2 # quadratic variance
        return self.mean, self.var

    def __repr__(self):
        return f'Mean: {self.mean.item():.5f} and var.: {self.var.item():.5f}'

#%%
def neg_log_density(y, mean, var):
    return torch.log(var) + (y - mean)**2 / var # ignoring sqrt(2pi) and 1/2

model = NormalModel()
opt = torch.optim.Adam(model.parameters(), lr=0.1)

#%%
max_epoch = 10000
for epoch in range(max_epoch):
    opt.zero_grad()
    # if epoch % 10 == 0 and max_epoch - epoch > 100:
    #     sample = torch.randint(N, (1, 100)).squeeze()
    # elif max_epoch - epoch == 100:
    #     sample = torch.arange(0, N)

    # mean, var = model(x[sample])
    # loss = neg_log_density(y[sample], mean, var).sum()
    mean, var = model(x)
    loss = neg_log_density(y, mean, var).sum()
    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        print(f'{epoch}:: Loss = {loss.item()}')

#%%
for p in model.parameters():
    print(p)
#%%
m, v = model(x)
s = torch.sqrt(v)
q90 = 1.64485 # -scipy.stats.norm.ppf(0.05)
cil = (m - q90 * s).detach().squeeze()
cir = (m + q90 * s).detach().squeeze()

plt.scatter(x, y, marker='.', c='grey')
plt.plot(x, rm, '-', c="green")
plt.plot(x, m.tolist(), '-', c='red')
plt.fill_between(x.squeeze(), cil, cir, color = 'orange', alpha=0.5)
plt.plot(x, rm+q90*rs,'-.', c='black')
plt.plot(x, rm-q90*rs,'-.', c='black')

plt.show()
