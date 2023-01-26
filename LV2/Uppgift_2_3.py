import torch
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)


# Uppgift 2
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class OurModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.q = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.u = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
    
    def  forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.p*self.q + self.u*(self.p + self.q - 1)
        external_grad = torch.tensor([1.])
        L.backward(gradient = external_grad)
        norm = torch.norm(torch.tensor(p.grad, q.grad, u.grad))
        return norm
