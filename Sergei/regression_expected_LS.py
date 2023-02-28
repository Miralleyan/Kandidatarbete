import numpy as np
import torch
from torch import nn
from torch_measure import TorchMeasure, MeasureMinimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

################# LINEAR REGRESSION ################
print('Linear regression alpha + x, where alpha is random, its distribution is to be estimated')
#  Model: -2 + x + Norm(0,sigma=2)
# create dummy data for training
N = 1000
X = torch.linspace(0, 10, N)
X = X.reshape(-1, 1) # making it a column-vector
Y1 = -2 + X + torch.from_numpy(np.random.normal(0,2,(N,1))).float()
# np.save('Y1.npy', Y1.detach().numpy())
# Y1 = torch.tensor(np.load('Y1.npy'))
# measure
amin = -5
amax = 3
no_atoms = 101
locations = np.linspace(amin, amax, no_atoms)
weights = np.ones(no_atoms) / no_atoms
alpha = TorchMeasure(locations, weights)


def ESSR(mes: TorchMeasure, x, y): # expected sum of squared residuals
    M = torch.add(mes.locations, x - y) # a_j + x_i - y_i : a matrix N x no_atoms
    EA = torch.matmul(torch.pow(M, 2), mes.weights)  # sum_j (a_j + x_i-y_i)^2 w_j = E (alpha + x_i-y_i)^2
    return EA.sum()

regression = MeasureMinimizer(alpha, ESSR, x=X, y=Y1)
regression.minimize(print_each_step=1)
print(f'Atoms: {regression.mes.atoms()}')
print(f'The support should be close to {(Y1 - X).mean():1.4f}. We have:')
print(f'Support of measure: {float(regression.mes.locations[regression.mes.support()]): 1.4f}')
print(f'Masses of atoms: {float(regression.mes.weights[regression.mes.support()]): 0.4f}')

# PS. Theoretically, since the loss function depends only on the second moment,
# the answer is a degenerated measure concentrated at the same value
# as the ordinary regression solution, i.e. close to the actual point -2.

# !! this is for 2 parameters estimation: alpha AND beta:
# from sklearn.linear_model import LinearRegression
# lin_model = LinearRegression()
# lin_model.fit(X.numpy(), Y1.numpy())


######### Using softmax and ordinary optimisation ##########
n_atoms = 101

model = nn.Sequential(
    nn.Linear(1, n_atoms, bias=False),
    # nn.Linear(n_atoms, n_atoms, bias=False),
    nn.Softmax(dim=0)
)

## Mimicing useful measure functions:
def support(weights, tol_supp=1e-3):
    """ Support of the measure up to tol_supp tolerance of what is considered as 0 """
    w = weights.detach().numpy()
    tol = np.abs(w).sum() * tol_supp
    return np.arange(len(w))[np.abs(w) > tol]

def print_measure(w, tol_supp=1e-6):
    print('location     weight')
    for i in support(w, tol_supp):
        print(f'{locations[i].item():1.3f}   {w[i].item(): 0.{int(-np.log10(tol_supp))}f}')

def weights():
    dummy = torch.ones(1, requires_grad=True)
    return model(dummy)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
locations = torch.linspace(amin, amax, no_atoms)

def ESSR2(x, y):  # expected sum of squared residuals
    M = torch.add(locations, x - y) # a_j + x_i - y_i : a matrix N x no_atoms
    EA = torch.matmul(torch.pow(M, 2), weights())  # sum_j (a_j + x_i-y_i)^2 w_j = E (alpha + x_i-y_i)^2
    return EA.sum()

def grad_is_zero(tolerance=1e-3):
    grad = []
    for param in model.parameters(): # grads of parameters on each layer of the model NN
        for i in range(len(param)):
            grad.append(param.grad[i].item())
    return np.array(grad).max() < tolerance

tol_supp = 1e-3
tol_grad = 1e-3
for i in range(10000):
    val = ESSR2(X, Y1) #
    if i % 100 == 99:
        print(f'Step {i+1}: Less = {np.round(val.item(),4)}')
    optimizer.zero_grad()
    val.backward()
    if grad_is_zero(tol_grad):
        print(f'Minimum is attained: gradient is 0 up to {tol_grad} accuracy')
        print_measure(weights(), tol_supp)
        break
    optimizer.step()

