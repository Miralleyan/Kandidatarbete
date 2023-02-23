import torch
from torch import nn
from torch_measure import TorchMeasure, MeasureMinimizer


device = "cuda" if torch.cuda.is_available() else "cpu"

# ### ------- Simple examples of autograd computation -------- ###
# # Computing the gradient for weights:
# wt = nn.Parameter(torch.tensor([0.2, 0.8]))
# var = -wt.prod()  # function of the parameters
# var.backward()  # gradient computation
# wt.grad  # here it is!
#
#
# # Same with a separately defined loss function with nn.Module:
# class Loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, weights):
#         return -weights.prod()
#
#
# loss = Loss()
# # bern_wt = nn.Parameter(torch.tensor([0.2, 0.8]))
# bern_wt = torch.tensor([0.2, 0.8],
#                        requires_grad=True)  # looks like "nn.Parameter" is not needed once Loss is nn.Module
# loss_cur = loss(bern_wt)
# loss_cur.backward()
# print(bern_wt.grad)
#
#
# # This works too without any reference to nn.Module:
# def loss0(wt):
#     return -wt.prod()
#
#
# bern0_wt = torch.tensor([0.2, 0.8],
#                         requires_grad=True)  # looks like "nn.Parameter" is not needed once Loss is nn.Module
# loss_cur = loss0(bern0_wt)
# loss_cur.backward()
# print(bern0_wt.grad)
#
#
# # Using function of measure:
# def loss(mes: TorchMeasure):
#     return -mes.weights.prod()
#
#
# bern = TorchMeasure([0, 1], [0.2, 0.8])
#
# for step in range(10):
#     loss_cur = loss(bern)  # compute the goal function
#     loss_cur.backward()  # compute gradient
#     print(f'----Step {step + 1}----')
#     print(bern)
#     print(f'grad = {bern.weights.grad}')
#     if bern.stop_criterion(bern.weights.grad):
#         print('Optimum is attained. The optimal measure:')
#         print(bern)
#         break
#     else:
#         bern.take_step(bern.weights.grad, 0.1)  # move to new measure
#         print('new measure:')
#         print(bern)
#         bern.weights.grad = None  # reset the gradient so that they do not accumulate
#
#
# ### ------- End of simple examples -------- ###

######################### Bernoulli example #####################
print('Optimisation of the variance of Bernoulli(p) distribution')
# goal function for minimisation:
def var(mes: TorchMeasure):
    return mes.weights.prod()
# goal function for maximisation:
def minus_var(mes: TorchMeasure):
    return -mes.weights.prod()

# Starting measure
bern = TorchMeasure([0, 1], [0.21, 0.79]).to(device)

# creating an instance for maximisation of variance:
opt = MeasureMinimizer(bern, minus_var).to(device)
print(f'Starting measure:\n{opt.mes}')
print(f'Goal function value: {opt.val}')
# print(opt.goal_fn(opt.mes)) # should be the same as opt.val
print(f'Gradient:\n{opt.grad}') # should be opt.mes swapped
print('Making one step with learning rate 0.1')
opt.step(lr=0.1) # take one step. Output: the learning rate and whether this step reduced the goal_fn
print(f'Current measure:\n{opt.mes}')
print(f'Current gradient:\n{opt.grad}')
print('\nRunning optimisation:')
opt.minimize(print_each_step=1)
print(f'Obtained measure:\n{opt.mes}') # solution is Bern(1/2)
print('decreasing tolerance of what is considered a constant for grad on support of mes:')
opt.minimize(silent=True, tol_const=1e-4)
print(f'Optimal measure:\n{opt.mes}')

# creating an instance for minimisation of variance:
opt = MeasureMinimizer(bern, var).to(device)
opt.minimize(silent=False, print_each_step=1)
print(f'Optimal measure:\n{opt.mes}') # solution is either Bern(0) or Bern(1)
print(opt.mes.atoms(0.01)) # atoms of mass at least 0.01


######### Optimisation with NN with softmax ##########
print('Minimizing the variance of Bernoulli distribution using NN with Softmax function')
n_atoms = 2

# Neural network:
model = nn.Sequential(
    nn.Linear(n_atoms, n_atoms, bias=False),
    nn.Softmax(dim=0)
)

def bernoulli_model():
    weights = torch.tensor([1., 2.], requires_grad=True) # initial weights
    return model(weights)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for i in range(1000):
    val = bernoulli_model()
    val = val.prod() # -variance
    if i % 100 == 99:
        print(f'Step {i+1}: variance is {val.item(): 0.6f}')
    optimizer.zero_grad()
    val.backward()
    optimizer.step()

bernoulli_model()

