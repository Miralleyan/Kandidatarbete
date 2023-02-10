import torch
from torch import nn
# inspired by
# https://www.youtube.com/watch?v=1wqKowfoUPM

## maximizing variance of Bernoulli distribution
p0 = torch.tensor(0.2, requires_grad=True) # initial value, initialises computation of grad
p = p0
optimizer = torch.optim.Adam([p], lr=0.1)
steps = 50
for step in range(steps):
    optimizer.zero_grad() # otherwise, the grads on steps will sum up
    var = - p*(1-p) # the minus so that to minimise var
    var.backward()
    optimizer.step()
    if step % 10 == 9:
        print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {-var.item(): 1.4f}')

print(f'Optimum found: p={p.detach(): 0.4f}')
# detach() is like item(), it also drops grad from tensor,
# but works on tensors with more than 1 element and also produces a tensor