import torch
import matplotlib.pyplot as plt
import numpy as np

class PytorchMeasure:
    def __init__(self, locations: torch.Tensor, weights: torch.Tensor):
        """
        Group 1: 
        Group 2: 
        """
        self.locations = torch.nn.parameter.Parameter(locations)#Input must be tensors
        self.weights = torch.nn.parameter.Parameter(weights)
    
    def __str__(self) -> str:
        """
        Returns the locations and weights of the measure as a string.
        :returns: str
        Responsibilty: Filip
        """
        return "Locations: " + self.locations.tolist().__str__() + "\nWeights: " + self.weights.tolist().__str__()

    def __repr__(self) -> str:
        """
        Returns the locations and weights of the measure as a string.
        :returns: str
        Responsibilty: Filip
        """
        return self.__str__()

    def total_mass(self) -> float:
        """
        Returns the sum of all weights in the measure: \sum_{i=1}^n w_i
        :returns: float
        Responsibility: Johan
        """
        return sum(self.weights).item()

    def total_variation(self) -> float:
        """
        Responsibility: Samuel
        Returns the sum of the absolute value of all weights in the measure: \sum_{i=1}^n |w_i|
        :returns: float
        """
        return sum(abs(self.weights)).item()

    def support(self):
        """
        Responsibility: Johan
        Returns all locations where the weights are non-zero
        """
        return self.locations[self.weights != 0]  # locations where weight is non-zero
        # add `.detach()` if dependency on self.locations and self.weights should be ignored when computing gradients
        # built-in torch functions are probably faster than python list comprehensions.

        # return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item()!=0])

    def positive_part(self):
        """
        Responsibility: Samuel
        Returns all locations where the weights are positive
        """
        return self.locations[self.weights > 0]
        # again `.detach()` if we don't want dependence on locations and weight when computing gradient on things depending
        # on `positive_part()`

        # return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item() > 0])

    def negative_part(self):
        """
        Responsibility: Johan
        Returns all locations where the weights are negative
        """
        return self.locations[self.weights < 0]
        #return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item()<0])

    def put_mass(self, mass, location_index) -> float:
        """
        In current form, this method puts mass at a specified location, s.t. the location still
        has mass less at most 1 and returns how much mass is left to distribute.
        :param: mass to put, index of location to put at
        :returns: mass left to add to measure after adding at specified location
        Responsibility: Johan, Samuel
        """
        with torch.no_grad():
            if self.weights[location_index].item() + mass > 1:
                mass_distributed = 1 - self.weights[location_index].item()
                self.weights[location_index] = 1
            else:
                self.weights[location_index] += mass
                mass_distributed = mass
        return mass_distributed

    def take_mass(self, mass, location_index) -> float:
        """
        Responsibility: Samuel
        In current form, this method takes mass from a specified location, s.t. the location still
        has non-negative mass and returns how much mass is left to take.
        :param: mass left to take, index of location to take at
        :returns: mass left to remove from measure after removing from specified location
        """
        with torch.no_grad():
            if mass > self.weights[location_index].item():
                mass_removed = self.weights[location_index].item()
                self.weights[location_index] = 0
            else:
                self.weights[location_index] -= mass
                mass_removed = mass
        return mass_removed



    def sample(self, size):
        """
        Responsibility: Samuel
        Returns a sample of numbers from the distribution given by the measure
        :param: size of wanted sample
        :returns: sample of random numbers based on measure
        """
        sampling = torch.multinomial(self.weights, size, replacement = True)
        sample = torch.tensor([self.locations[element.item()] for element in sampling])
        return sample

    def step(self, loss_fn, lr):
        """
        Responsibility: Hampus
        Steepest decent with fixed total mass using provided loss function and learning rate

        :param loss_fn: reference to loss function using model weights as input, ex. loss_fn(weights)
        :param lr: learning rate
        """
        # if lr is too high, adjust to the highest possible value
        if lr/2 >= len(self.weights) - 1:
            lr = 2 * len(self.weights) - 2.01

        # Zero gradient
        self.weights.grad = torch.zeros(len(self.weights))

        # Compute gradient
        loss_fn(self.weights).backward()
        grad_abs = torch.argsort(self.weights.grad)

        # Distribute positive mass
        mass_pos = lr/2
        i = 0
        while mass_pos != 0:
            index = grad_abs[i].item()
            mass_pos -= self.put_mass(mass_pos, index)
            i += 1

        # Distribute negative mass
        mass_neg = lr/2
        i = -1
        while mass_neg != 0:
            index = grad_abs[i].item()
            mass_neg -= self.take_mass(mass_neg, index)
            i -= 1

    def visualize(self):
        """
        Responsibility: Karl
        Visualization of the weights
        """
        plt.bar(self.locations.tolist(), self.weights.tolist())
        plt.show()

        # Ensure sum = 1
        torch.softmax(self.weights, dim=0)

def main():
    a = torch.tensor([-0.1, 0.1, 0.3, 0.1, 0.4])
    b = torch.tensor([1., 2., 3., 4., 5.])

    c = PytorchMeasure(b, a)
    #print(c.negative_part())
    #print(c.put_mass(0.9, 1))
    #print(c)
    test_sample()


def test_sample():
    a=torch.tensor([0.1, 0.1, 0.3, 0.1, 0.4])
    b=torch.tensor([1., 2., 3., 4., 5.])

    c=PytorchMeasure(b, a)
    c.visualize()

    print(c.sample(2000))


if __name__ == "__main__":
    main()