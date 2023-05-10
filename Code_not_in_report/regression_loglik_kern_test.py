import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_measure as pm

# NN with softmax
N = 1000
X = torch.linspace(0, 10, N)
X = X.reshape(-1, 1) # making it a column-vector
sigma = 1 # st.dev. of the noise
Y1 = -2. + X + torch.from_numpy(np.random.normal(0,sigma,(N,1))).float()
no_atoms = 20

model = torch.nn.Sequential(
    torch.nn.Linear(no_atoms, no_atoms, bias=False),
    torch.nn.Softmax(dim=0)
)

def lin_model(a,x):
    return a+x

def measure_model():
    weights = torch.tensor([1.]*no_atoms, requires_grad=True)
    return model(weights)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
def loss():
    

for i in range(1000):
    alpha = measure_model()
    print(alpha)
    val = loss()
    if i % 100 == 99:
        print(f'Step {i+1}: loss is {loss.item(): 0.6f}')
    optimizer.zero_grad()
    loss.backward()
    #print(alpha.weights.grad)
    optimizer.step()
