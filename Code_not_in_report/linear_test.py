import torch
import numpy as np
import matplotlib.pyplot as plt
import linear_combination_optimizer as lco

length = 500

x = torch.linspace(-5, 5, length)
y = np.load(f'../Finalized/test_data/data_{length}_y_sqr_{24}.npy')
opt = lco.Optimizer(x, y, order=3)
mu, sigma = opt.optimize(epochs=300, test=False)


mu = np.array(mu)
sigma = np.array(sigma)
plt.scatter(x, y, marker='.', label='data', zorder=0)
plt.plot(x, mu, 'r', label='mean', zorder=5)
plt.fill_between(x.squeeze(), (mu-sigma), (mu+sigma), alpha=0.5, label='1std', zorder=3)
plt.fill_between(x.squeeze(), (mu-sigma*2), (mu+sigma*2), alpha=0.5, label='2std', zorder=2)
plt.legend()
#plt.savefig('myplot.png', dpi=300)
plt.show()
