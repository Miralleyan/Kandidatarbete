import pytorch_measure as pm
import torch
# goal function for minimisation:
def var(listmes: pm.Measure):
    return listmes[0].weights.prod()
# goal function for maximisation:
def minus_var(listmes: pm.Measure):
    return -listmes[0].weights.prod()

# Starting measure
bern = pm.Measure(torch.tensor([0, 1]), torch.tensor([0.21, 0.79]))
opt = pm.Optimizer(bern)
opt.minimize(minus_var, verbose = True)
opt = pm.Optimizer(bern)
opt.minimize(var, verbose = True)

print("----- < using softmax instead: > -----")

import torch
import matplotlib.pyplot as plt



m = torch.nn.Sequential(
    torch.nn.Linear(1, 2),
    torch.nn.Softmax(0)
)
loss_fn = lambda y_pred, y: y_pred.prod(0)
opt = torch.optim.Adam(m.parameters(), lr=0.1)

for epoch in range(2000):
    x = torch.tensor([0.], dtype=torch.float32)
    y_pred = m(x)
    
    loss = loss_fn(y_pred, "placeholder")
    opt.zero_grad()

    loss.backward()
    
    opt.step()

    if epoch % 50 == 0:
        print(epoch, f"\tloss={loss.item()}")


