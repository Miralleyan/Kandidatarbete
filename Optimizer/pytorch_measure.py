import torch
import matplotlib.pyplot as plt
import numpy as np


class Measure:
    def __init__(self, locations: torch.nn.parameter, weights: torch.nn.parameter):
        self.locations = locations
        self.weights = weights

    def __str__(self) -> str:
        """
        Returns the locations and weights of the measure as a string.
        :returns: str
        Responsibilty: Filip, Karl
        """

        out = "\033[4mLocations:\033[0m     \033[4mWeights:\033[0m \n"
        for i in range(len(self.weights)):
            out += f'{self.locations[i].item(): < 10}     {self.weights[i].item(): < 10}\n'
        return out

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

    def support(self, tol_supp = 1e-12):
        """
        Responsibility: Johan
        Returns all locations where the weights are non-zero
        """
        tol = self.total_variation*tol_supp
        return self.locations[self.weights > tol]  # locations where weight is non-zero
        # add `.detach()` if dependency on self.locations and self.weights should be ignored when computing gradients
        # built-in torch functions are probably faster than python list comprehensions.

    def positive_part(self):
        """
        Responsibility: Samuel
        Returns the positive part of the Lebesgue decomposition of the measure
        """
        return Measure(self.locations, torch.max(self.weights, torch.zeros(self.weights.size)))

    def negative_part(self):
        """
        Responsibility: Johan
        Returns the negative part of the Lebesgue decomposition of the measure
        """
        return Measure(self.locations, torch.min(self.weights, torch.zeros(self.weights.size)))


    def sample(self, size):
        """
        Responsibility: Samuel
        Returns a sample of numbers from the distribution given by the measure
        :param: size of wanted sample
        :returns: sample of random numbers based on measure
        """
        if torch.any(self.weights < 0):
            assert ValueError("You can't have negative weighs in a probability measure!")

        sampling = torch.multinomial(self.weights, size, replacement = True)
        sample = torch.tensor([self.locations[element.item()] for element in sampling])
        return sample

    def zero_gradient(self):
        self.weights.grad = torch.zeros(len(self.weights))



    def visualize(self):
        """
        Responsibility: Karl
        Visualization of the weights
        """
        plt.bar(self.locations.tolist(), self.weights.tolist(), width=0.1)
        plt.axhline(y=0, c="grey", linewidth=0.5)
        plt.show()


class Optimizer:

    def __init__(self, weights: torch.nn.parameter, ):
        self.weights = weights

    def put_mass(self, mass, location_index) -> float:
        """
        In current form, this method puts mass at a specified location, s.t. the location still
        has mass less at most 1 and returns how much mass is left to distribute.
        :param: mass to put, index of location to put at
        :returns: mass left to add to measure after adding at specified location
        Responsibility: Johan, Samuel
        """
        with torch.no_grad():
            self.weights[location_index] += mass


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


    def step(self, loss_fn, lr):
        """
        Responsibility: Hampus
        Steepest decent with fixed total mass

        :param lr: learning rate
        """
        ''' Gör externt ist
        # set global lr if not already set (lite temp för att testa minskande lr)
        if not self.learning_rate:
            self.learning_rate = lr
        else:
            lr = self.learning_rate

        # if lr is too high, adjust to the highest possible value
        if lr/2 >= len(self.weights) - 1:
            lr = 2 * len(self.weights) - 2.01
        '''

        # Sort gradient
        grad_sorted = torch.argsort(self.weights.grad)

        # Distribute positive mass
        mass_pos = lr
        i = 0
        while mass_pos != 0:
            index = grad_sorted[i].item()
            mass_pos -= self.put_mass(mass_pos, index)
            i += 1

        # Distribute negative mass
        mass_neg = lr
        i = -1
        while mass_neg != 0:
            index = grad_sorted[i].item()
            mass_neg -= self.take_mass(mass_neg, index)
            i -= 1

def main():
    a = torch.tensor([-0.1, 0.1, 0.3, 0.1, 0.4])
    b = torch.tensor([1., 2., 3., 4., 5.])

    c = Measure(b, a)
    #print(c.negative_part())
    #print(c.put_mass(0.9, 1))
    print(c)
    test_sample()


def test_sample():
    a=torch.tensor([0.1, 0.1, 0.3, 0.1, 0.4])
    b=torch.tensor([1., 2., 3., 4., 5.])

    d=Measure(b, a)
    
    d.visualize()

    print(d.sample(2000))


if __name__ == "__main__":
    main()