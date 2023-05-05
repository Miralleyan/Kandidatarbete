import torch
from torch import nn, optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt


df = pd.read_csv('boston_housing.csv')

data = df.to_numpy()

x = data[:, :-1]
y = data[:, -1]

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# normalize data to enable better prediction
x -= x.mean(0)
x /= x.std(0)

class GaussianRegression(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.pred_layer = nn.Linear(input_features, 1)
        self.std = nn.Parameter(torch.ones(input_features, requires_grad=True))
        self.std_default = nn.Parameter(torch.tensor(1., requires_grad=True))
    
    def forward(self, x):
        y_pred = self.pred_layer(x)
        var_pred = (x.pow(2) * self.std.pow(2)).sum(1) + self.std_default.pow(2)
        return y_pred, var_pred


model = GaussianRegression(13)
opt = optim.SGD(model.parameters(), lr=0.02)
criterion = nn.GaussianNLLLoss(full=True)

train_set, val_set = torch.utils.data.random_split(x, [400, 106])

dataset = TensorDataset(x[train_set.indices], y[train_set.indices])
loader = DataLoader(dataset, batch_size=200, shuffle=True)


for epoch in range(10000):
    for x_sample, y_sample in loader:
        opt.zero_grad()

        y_pred, var_pred = model(x_sample)

        loss = criterion(y_pred, y_sample, var_pred)
        loss.backward()
        opt.step()
    
    if epoch % 100 == 99 or epoch < 10:
        print(f"{epoch + 1}: {loss.item()}")

y_pred, var_pred = model(x[val_set.indices])
print(f"mean prediction parameters: {list(model.pred_layer.parameters())}")
print(f"variance prediction: {1 * model.std.pow(2)}")
print(f"variance offset: {model.std_default ** 2}")

y_np = y[val_set.indices].squeeze().detach().numpy()
y_pred_np = y_pred.squeeze().detach().numpy()

plt.scatter(y_np, y_pred_np)
plt.plot([0, 50], [0, 50])
plt.ylabel('predicted price')
plt.xlabel('actual price')
plt.show()

hits = ((y[val_set.indices] - y_pred).abs().squeeze() < 1.96 * var_pred.sqrt()).sum()
proportion = hits / 106
print(f"proportion of points within 95% interval: {proportion}")
