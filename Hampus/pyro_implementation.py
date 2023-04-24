import pyro
import torch
from pyro.distributions import Normal
from pyro.distributions import HalfNormal
from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as opt


def model(x, y):
    w = pyro.sample("w", Normal(0., 1.))
    b = pyro.sample("b", Normal(0., 1.))
    mu = w * x + b
    with pyro.plate("data", len(x)):
        pyro.sample("obs", mu, obs=y)


def guide(x, y):
    w_loc = pyro.param("w_loc", torch.tensor(0.))
    w_scale = pyro.param("w_scale", torch.tensor(1.), constraint=constraints.positive)
    b_loc = pyro.param("b_loc", torch.tensor(0.))
    b_scale = pyro.param("b_scale", torch.tensor(1.), constraint=constraints.positive)

    w = pyro.sample("w", Normal(w_loc, w_scale))
    b = pyro.sample("b", Normal(b_loc, b_scale))


# generate some data
x = torch.linspace(0, 10, 1000)
y = 3 * x + 1 + 0.2 * torch.randn(1000)

# set up the optimizer and inference algorithm
optim = opt.Adam({'lr': 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

# train the model
for epoch in range(1000):
    svi.step(x, y)
    if epoch % 100 == 0:
        w = pyro.param("w_loc").item()
        b = pyro.param("b_loc").item()
        sigma = pyro.param("sigma_loc").item()
        print(f"{epoch}:: w: {w:.2f}, b: {b:.2f}, sigma: {sigma:.2f}")

# extract the learned parameters
w = pyro.param("w_loc").item()
b = pyro.param("b_loc").item()
sigma = pyro.param("sigma_loc").item()

print(f"w: {w:.2f}, b: {b:.2f}, sigma: {sigma:.2f}")