import torch
import numpy as np
from matplotlib import pyplot as plt

# Introduce slack variables, so that all constraints become equalities
# Slack variables are basically non-negative variables that get added to the 
# inequality constraints, so that they become equalities
pars = torch.nn.Parameter(torch.tensor(0.5).repeat(11))

opt = torch.optim.Adam(params=[pars], lr=0.1)

for epoch in range(100):
    opt.zero_grad()
    print(pars)
    # Clamp on slack variables
    with torch.no_grad():
        pars[7:].clamp_(0)
    # Calculate Lagrangian with slack variables
    L = pars[0]*pars[1] + pars[2]*(pars[0] + pars[1] - 1) +\
        pars[3]*(-pars[0] + pars[7]) + pars[4]*(pars[0] - 1 + pars[8]) +\
        pars[5]*(-pars[1] + pars[9]) + pars[6]*(pars[1] - 1 + pars[10])
    # Take the gradient
    L.backward()
    parGrad = torch.nn.Parameter(pars.grad)
    # Define the loss function as the squared norm of the gradient
    loss = parGrad[0]**2 + parGrad[1]**2 + parGrad[2]**2 +\
        parGrad[3]**2 + parGrad[4]**2 + parGrad[5]**2 +\
        parGrad[6]**2 + parGrad[7]**2 + parGrad[8]**2 +\
        parGrad[9]**2 + parGrad[10]**2
    # Take the gradient of the loss function
    loss.backward()
    opt.step()

pars = pars.detach()
pars[7:].clamp_(0)
print(pars)

plt.plot(np.linspace(0,1), np.ones(50)-np.linspace(0,1))
plt.plot(pars[0], pars[1], ".")
plt.show()