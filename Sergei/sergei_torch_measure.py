import numpy as np
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class TorchMeasure(nn.Module):
    def __init__(self, locations, weights, optimize_locations = False):
        super(TorchMeasure, self).__init__()
        assert len(locations) == len(weights), 'locations and weights should have the same length!'
        if not optimize_locations:
            self.locations = torch.tensor(locations)
        else:
            self.locations = torch.tensor(locations, dtype=torch.float64, requires_grad=True)
        self.weights = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
        self.n = len(weights)

    def __str__(self):
        out = 'locations weights\n'
        for i in range(self.n):
            out += f'{self.locations[i].item()}      {self.weights[i].item()}\n'
        return out

    def __repr__(self):
        out = 'locations weights\n'
        for i in range(self.n):
            out += f'{self.locations[i].item()}      {self.weights[i].item(): 0.6f}\n'
        return out

    def total_variation(self):
        """ Total variation of the measure"""
        return float(torch.abs(self.weights).sum())

    def total_mass(self):
        """ Total mass of the measure"""
        return float(self.weights.sum())

    def support(self, tol_supp=1e-12):
        """ Support of the measure up to tol_supp tolerance of what is considered as 0 """
        tol = self.total_variation() * tol_supp
        return np.arange(self.n)[torch.abs(self.weights) > tol]

    def is_positive(self, tol_pos=1e-6):
        """ Check if the measure is non-negative"""
        tol = self.total_variation() * tol_pos
        return bool(torch.all(self.weights >= -tol))

    def is_probability(self, tol=1e-6):
        """ Check whether this is probability measure """
        return self.is_positive() and np.abs(self.total_mass() - 1) <= tol

    def copy(self):
        """ Create a copy of the measure"""
        return TorchMeasure(self.locations.detach().numpy(), self.weights.detach().numpy())

    def sample_from(self, nmb):
        """ sample n locations from probability distribution proportional to the measure """
        assert self.is_positive(), 'measure should be positive to sample from it!'
        cdf = np.cumsum(self.weights.detach().numpy())/self.total_mass()
        sample = []
        for i in range(nmb):
            sample.append(self.locations.detach().numpy()[cdf > np.random.random()][0]) # inverse of the cdf method
        return np.array(sample)

    def put_mass(self, grad: torch.Tensor, epsilon: float):
        """ Put mass epsilon to the coordinate of weights where grad is minimal """
        with torch.no_grad():
            self.weights[grad.argmin()] += epsilon
        ## without "with" above autograd gets confused when tensor with gradient is changed in place
        # see https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf


    def take_mass(self, grad: torch.Tensor, epsilon: float):
        """ Take mass epsilon from the coordinates of weights where grad is maximal starting from the maximal, next to maximal etc """
        decreasing_grad_idx = torch.flip(grad.argsort(), dims=[0])
        mass_left = epsilon
        with torch.no_grad():
            for idx in decreasing_grad_idx:
                self.weights[idx] -= mass_left
                if self.weights[idx] < 0.:
                    mass_left = -self.weights[idx]
                    self.weights[idx] = 0.
                else:
                    break

    def take_step(self, grad: torch.Tensor, epsilon: float, silent=True):
        """ true gradient descent  preserving the total mass of weights"""
        self.put_mass(grad, epsilon)
        self.take_mass(grad, epsilon)
        if not silent:
            return print(self)

    def stop_criterion(self, grad, tol_supp=1e-6, tol_const=1e-3, adaptive=False):
        """
        Grad should be minimal and constant on the support of the measure.
        Consider it to be a constant if it varies by less than tol_const.
        If adaptive = True, if it varies less than tol_const * (range of grad)
        """
        min_grad, max_grad = grad.min(), grad.max()
        if adaptive:
            tol_const *= (max_grad - min_grad)
        return bool(grad[self.support(tol_supp)].max() - min_grad < tol_const)
