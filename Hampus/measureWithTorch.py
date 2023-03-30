import torch
import numpy as np
import matplotlib.pyplot as plt


class NormalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Parameter(torch.randn(()))
        self.s = torch.nn.Parameter(torch.randn(()))

    def __str__(self):
        return f'Mean: {self.m.item():.4f} and Std: {self.s.item():.4f}'




N = 1000 # datapoints

x = torch.linspace(0, 10, N)
y = torch.randn(N)

model = NormalModel()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    loss = -(1 / np.sqrt(2 * np.pi) * torch.exp(-(y-model.m) / (model.s ** 2 / 2) / model.s)/model.s).log().sum()

    if epoch % 10 == 0:
        print(f'{epoch}:: Loss = {loss.item()}')

    loss.backward()
    opt.zero_grad()
    opt.step()

print(model)
