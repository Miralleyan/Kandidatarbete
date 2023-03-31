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


N = 200  # datapoints

x = torch.linspace(1, 5, N).view(-1, 1)
y = (x)**2 -6*x + torch.tensor(np.random.normal(2., 0.2, N), dtype=torch.float).view(-1, 1)

model = NormalModel()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

for epoch in range(1000):
    opt.zero_grad()
    mean, std = model(x)
    dist = torch.distributions.Normal(mean, std)
    loss = -dist.log_prob(y).sum()
    #kern = 1 / np.sqrt(2 * np.pi) * torch.exp(-(y- mean) ** 2 / 2)
    #loss = loss_fn(mean, y)

    if epoch % 50 == 0:
        print(f'{epoch}:: Loss = {loss.item()}')
    loss.backward()
    opt.step()

x_plot = torch.linspace(1.05, 4.96, 40).view(-1, 1)
mean, std = model(x_plot)
mean = mean.detach().squeeze(-1)
std = std.detach().squeeze(-1)
plt.scatter(x, y, s=3)
plt.errorbar(x=x_plot, y=mean, yerr=std)
#plt.plot(x_plot, mean.detach())
plt.show()

