import torch


class LinearRegression(torch.nn.Module):
    def __init__(self, a=None, b=None):
        super().__init__()

        if a is None:
            self.a = torch.nn.Parameter(torch.randn(()))
        if b is None:
            self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x

    def string(self):
        return f'y = {self.a.item():.4f} + {self.b.item():.4f} * x'


class Polynom(torch.nn.Module):
    order = 0

    def __init__(self, order: int):
        super().__init__()
        self.order = order
        self.linear = torch.nn.Linear(order, 1)
        self.flatten = torch.nn.Flatten(0, 1)

    def forward(self, x, device):
        p = torch.tensor([i for i in range(1, self.order + 1)]).to(device)
        xx = x.unsqueeze(-1).pow(p)
        output = self.linear(xx)
        output = self.flatten(output)
        return output
