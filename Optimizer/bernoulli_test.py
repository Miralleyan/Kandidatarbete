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
