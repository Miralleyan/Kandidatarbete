import torch
import matplotlib.pyplot as plt

eq_line = torch.tensor([1., 1.])

θ = torch.tensor([0.2, 0.8], requires_grad=True)
optimizer = torch.optim.SGD([θ], lr=0.2)
for i in range(100):
    optimizer.zero_grad()
    output = torch.prod(θ) # = p * q
    output.backward()
    # enforce that θ sums to 1 through projecting gradient on eq_line.
    θ.grad -= eq_line * torch.dot(θ.grad, eq_line) / torch.dot(eq_line, eq_line)
    # decrese lr if we try to step outside allowed area:
    if torch.any(θ - θ.grad * optimizer.param_groups[0]['lr'] < 0):
        optimizer.param_groups[0]['lr'] *= 0.7
    else:
        optimizer.step()
    print(θ)