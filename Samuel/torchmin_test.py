import torch
from torchmin import minimize_constr, minimize

# 1 variable case
fn = lambda p: p*(1-p)  # Define objective function to minimize
x0 = torch.tensor([0.6])    # Set initial value

# Minimize the function with constraints
res = minimize_constr(
    fn, x0, 
    max_iter=100,
    constr=dict(
        fun=lambda x: 2*x**1-1, 
        lb=1, ub=1
    ),
    bounds = dict(lb = 0, ub = 1),
    disp=3
)
print(res)  # Print results

# 2 variable case
fn = lambda p: p[0]*p[1]

res = minimize_constr(
    fn, x0, 
    max_iter=100,
    constr=dict(
        fun=lambda p: p[0]**1+p[1]**1, 
        lb=torch.tensor([1, 1]), ub=torch.tensor([1, 1])
    ),
    bounds = dict(lb = torch.tensor([0, 0]), ub = torch.tensor([1, 1])),
    disp=3
)
print(res)