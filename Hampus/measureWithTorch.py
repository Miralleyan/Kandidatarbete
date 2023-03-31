import torch
import numpy as np
import matplotlib.pyplot as plt


class NormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mean = None
        self.std = None
        k = 12

        self.hidden_layers = torch.nn.Sequential(
            torch.nn.Linear(1, k),
            torch.nn.Sigmoid(),
        )

        self.mean_layer = torch.nn.Linear(k, 1)
        self.std_layer = torch.nn.Linear(k, 1)
        self.elu = torch.nn.ELU(1)

    def forward(self, x):
        hl = self.hidden_layers(x)
        self.mean = self.mean_layer(hl)
        s = self.std_layer(hl)
        self.std = self.elu(s) + 1.

        return self.mean, self.std

    def __repr__(self):
        return f'Mean: {self.mean.item():.5f} and Std: {self.std.item():.5f}'


def f(a, x):
    return a


def k(mean, std, x):
    return 1 / np.sqrt(2 * np.pi) * torch.exp(-((x - mean)/std) ** 2 / 2) / std


N = 200  # datapoints
x = torch.linspace(-3, 5, N).view(-1, 1)
y = f(torch.tensor(np.random.normal(0, 2, N), dtype=torch.float).view(-1, 1), x)

mean = torch.tensor(torch.randn(()), requires_grad=True)
std = torch.tensor(torch.randn(()), requires_grad=True)

opt = torch.optim.Adam([mean, std], lr=0.1)
for epoch in range(1000):
    opt.zero_grad()
    kern = k(mean, std, y)
    loss = -kern.log().sum()

    #mean, std = model(x)
    #dist = torch.distributions.Normal(mean, std)
    #loss = -dist.log_prob(y).sum()
    #loss = loss_fn(mean, y)

    if epoch % 1000 == 0:
        print(f'{epoch}:: Loss = {loss.item()}')
    loss.backward()
    opt.step()

x_plot = torch.linspace(-3, 3, 100).view(-1, 1)
m = k(mean, std, x_plot).detach()
plt.plot(x_plot, m.abs())
plt.show()
print(f'Mean: {mean.item():.4f} and Std: {std.abs().item():.4f}')

