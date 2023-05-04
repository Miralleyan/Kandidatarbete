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

# # normalize data
# x -= x.mean(axis=0)
# x[:, :-1] /= x[:, :-1].std(axis=0)

class GaussianRegression(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.pred_layer = nn.Linear(input_features, 1)
        self.dist_vars_log = nn.Parameter(torch.zeros(input_features, requires_grad=True)) # want to force variance to be greater than zero.
        self.std_default = nn.Parameter(torch.tensor(1., requires_grad=True))
        #self.var_layer = nn.Linear(input_features, 1)
    
    def forward(self, x):
        y_pred = self.pred_layer(x)
        var_pred = (x.pow(2) * self.dist_vars_log.exp()).sum(1) + self.std_default.pow(2)
        return y_pred, var_pred


model = GaussianRegression(13)
opt = optim.SGD(model.parameters(), lr=0.02)
criterion = nn.GaussianNLLLoss()

prep_opt = optim.Adam(model.pred_layer.parameters(), lr=0.1)
prep_criterion = nn.MSELoss()

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

for epoch in range(5000):
    prep_opt.zero_grad()
    y_pred, _ = model(x)
    loss = prep_criterion(y_pred, y)
    loss.backward()
    prep_opt.step()

print(list(model.pred_layer.parameters()))
plt.scatter(y.detach().numpy(), y_pred.detach().numpy())
plt.show()


for epoch in range(20000):
    for x_sample, y_sample in loader:
        opt.zero_grad()

        y_pred, var_pred = model(x_sample)

        loss = criterion(y_pred, y_sample, var_pred)
        loss.backward()
        opt.step()
    
    if epoch % 100 == 99 or epoch < 10:
        print(f"{epoch + 1}: {loss.item()}")

#print(f"mean: {mean.detach().numpy()}")
#print(f"std: {(1 / var_inv).sqrt().detach().numpy()}")

y_pred, _ = model(x)
print(list(model.parameters()))

plt.scatter(y.detach().numpy(), y_pred.detach().numpy())
plt.plot([0, 50], [0, 50])
plt.show()
