import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_measure import TorchMeasure, MeasureMinimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

################# LINEAR REGRESSION ################
print('Linear regression alpha + x, where alpha is random, its distribution is to be estimated')


########## Likelihood maximisation ##########
torch.manual_seed(1)
#  Model: -2 + x + Norm(0,sigma=1)
# create dummy data for training
N = 40
X = torch.linspace(0, 10, N)
X = X.reshape(-1, 1) # making it a column-vector
Y1 = -2 + X + torch.from_numpy(np.random.normal(0,1,(N,1))).float()
# np.save('Y1.npy', Y1.detach().numpy())
# Y1 = torch.tensor(np.load('Y1.npy'))

amin = -5
amax = 3
no_atoms = 5
locations = np.linspace(amin, amax, no_atoms)
weights = np.array(np.random.uniform(0, 1, size=no_atoms))
# weights = np.ones(no_atoms)
weights = weights / weights.sum()
alpha = TorchMeasure(locations, weights) # random starting measure

def regression_model(a,x):
    return a+x

### Locations a_j of the measure are mapped by the model into a_j+x for each x
### So the likelihood of response y_i may be taken to equal the weight of the atom a_j(i)
### such that a_j(i) + x_i is closest to y_i
### These weights are fitted by minimizing minus log-likelihood

## preprocessing closest points:
# Compute for each x_i the index j(i) of the closest to y_i location a_j + x_i
closest_idx = []
for i in range(N):
    ab = (regression_model(alpha.locations, X[i]) - Y1[i]).abs()
    closest_idx.append(torch.argmin(ab))


def loglik(mes):  # - log-likelihood
    ''' Locations a_j of alpha are mapped by model to a_j + x_i for each x_i.
    Assign to corresponding y_i likelihood = the weight w_j(i) of a_j(i) such that
     a_j(i) + x_i is the closest to y_j.'''
    return -(mes.weights[closest_idx]).log().sum()

likmax = MeasureMinimizer(alpha, loglik, learning_rate=0.1)
likmax.minimize(print_each_step=1, silent=False, max_no_steps=1000)
#likmax.minimize(print_each_step=10, tol_const=0.01, adaptive=True)

# print(likmax.grad)
# likmax.mes.visualize()
locations_grid_size = (amax - amin) / (no_atoms - 1)
bins=torch.linspace(amin - locations_grid_size / 2, amax + locations_grid_size / 2, no_atoms + 1)
h,b = torch.histogram(Y1 - X, bins=bins)
midpoints = (b[1:] + b[:-1]) / 2

# side-by-side histogram. Can it be done smarter?
plt.bar(midpoints - 0.1, likmax.mes.weights.tolist(), width = 0.2, label='fitted')
plt.bar(midpoints + 0.1, h / h.sum(), width = 0.2, label='data')
plt.legend(loc='upper right')
plt.show()

