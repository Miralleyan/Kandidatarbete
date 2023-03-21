import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_measure import TorchMeasure, MeasureMinimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

################# LINEAR REGRESSION ################
print('Linear regression alpha + x, where alpha is random, its distribution is to be estimated')

########## Likelihood maximisation ##########

#  Model: -2 + x + Norm(0,sigma=1)
# create dummy data for training
N = 1000
X = torch.linspace(0, 10, N)
X = X.reshape(-1, 1)  # making it a column-vector
sigma = 1  # st.dev. of the noise
Y1 = -2. + X + torch.from_numpy(np.random.normal(0, sigma, (N, 1))).float()
# np.save('Y1.npy', Y1.detach().numpy())
# Y1 = torch.tensor(np.load('Y1.npy'))

# Measure - the initial distribution of alpha discretised to a grid:
amin = -5
amax = 3
no_atoms = 100
locations = np.linspace(amin, amax, no_atoms)
weights = np.array(np.random.uniform(0, 1, size=no_atoms))
# weights = np.ones(no_atoms)
weights = weights / weights.sum()
alpha = TorchMeasure(locations, weights)  # random starting measure - both locations and weights are row vectors
# scale for KDE below:
atoms_spacing = (amax - amin) / (no_atoms - 1)


## The model:
def model(a, x):
    return a + x


## Kernels for KDE
def gaussian_kernel(x: float, scale: float = atoms_spacing):
    return np.exp(-0.5 * x ** 2 / scale ** 2) / (np.sqrt(2 * np.pi) * scale)


def uniform(x: float, scale: float = 0.5):
    if abs(x) <= scale:
        return 0.5 / scale
    else:
        return 1e-12


uniform_kernel = np.vectorize(uniform)


def bartlett(x: float, scale: float = 0.5):
    out = (1 - np.abs(x / scale)) / scale
    if out <= 0.:
        return 1e-12
    else:
        return out


bartlett_kernel = np.vectorize(bartlett)


def kernel(x):
    return gaussian_kernel(x, atoms_spacing)


### Locations a_j of the measure are mapped by the model into a_j+x for each x
### The likelihood of response y_i is estimated via KDE from the images of atoms a_j(i)
### under the model applied: i.e. a_j + x_i
### These weights are fitted by minimizing minus log-likelihood

## preprocessing KDE:
# Compute kernel(y_i - model(a_j, x_i)) for each x_i and a_j
K = kernel(Y1 - model(alpha.locations, X))  # (N x no_atoms) tensor


def loglik(mes):  # - log-likelihood
    ''' Locations a_j of alpha are mapped to model(a_j, x_i) for each x_i.
    Assign to corresponding y_i likelihood = sum_j w_j(i) kernel(y_j-model(a_j,x_i)
    '''
    return -torch.tensordot(K, mes.weights, dims=1).log().sum()


likmax = MeasureMinimizer(alpha, loglik)
likmax.minimize(print_each_step=10)
likmax.minimize(tol_const=0.01, adaptive=True, silent=True, max_no_steps=10000)
likmax.visualize()


# producing histograms
def make_hist(n_bins):
    bins = np.linspace(amin - atoms_spacing / 2, amax + atoms_spacing / 2, n_bins + 1)
    bins_width = bins[1] - bins[0]
    h, b = np.histogram(Y1 - X, bins=bins)
    midpoints = (b[1:] + b[:-1]) / 2
    # measure weights of all the bins:
    mes_hist = np.zeros(n_bins)
    for i in range(n_bins):
        mes_hist[i] = likmax.mes.weights[
            (likmax.mes.locations > bins[i]) & (likmax.mes.locations <= bins[i + 1])].sum().tolist()

    # side-by-side histogram. Can it be done smarter?
    plt.bar(midpoints - 0.1 * bins_width, mes_hist, width=0.8 * bins_width, color='b', label='fitted')
    plt.bar(midpoints + 0.1 * bins_width, h / h.sum(), width=0.8 * bins_width, label='data', alpha=0.7, color='orange')
    plt.legend(loc='upper right')
    plt.show()

# make_hist(10)
