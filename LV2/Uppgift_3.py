import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)


# Task 2
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

p = torch.tensor(0.2, requires_grad=True)
q = torch.tensor(0.2, requires_grad=True)
u1 = torch.tensor(0.2, requires_grad=True)
u2 = torch.tensor(0.2, requires_grad=True)
u3 = torch.tensor(0.2, requires_grad=True)
u4 = torch.tensor(0.2, requires_grad=True)
u5 = torch.tensor(0.2, requires_grad=True)


def CreateLossFunc(func):
    pass





