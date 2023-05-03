import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt


df = pd.read_csv('boston_housing.csv')

data = df.to_numpy()

x = np.concatenate((data[:, :-1], np.ones((506, 1))), axis=1)
y = data[:, -1]

x = torch.tensor(x)
y = torch.tensor(y)

# # normalize data
# x -= x.mean(axis=0)
# x[:, :-1] /= x[:, :-1].std(axis=0)

mean = torch.zeros((14, 1), requires_grad=True, dtype=torch.float64) # 13 features, 1 constant
std = torch.ones((14, 1), requires_grad=True, dtype=torch.float64)

opt = torch.optim.SGD([mean, std], lr=0.1)

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=64)

for epoch in range(1000):
    for x_sample, y_sample in loader:
        opt.zero_grad()

        # dist = Normal(x_sample.matmul(mean), x_sample.pow(2).matmul(std.pow(2)).sqrt())
        dist = Normal(x.matmul(mean), 0.1 * torch.ones((506, 1), dtype=torch.float64))
        loss = -dist.log_prob(y_sample).sum()
        loss.backward()
        opt.step()
    
    if epoch % 5 == 4 or epoch < 10:
        print(f"{epoch + 1}: {loss.item()}")

plt.scatter(y.detach().numpy(), x.matmul(mean).detach().numpy())
plt.show()
