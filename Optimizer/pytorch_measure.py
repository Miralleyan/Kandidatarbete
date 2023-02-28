import torch
import matplotlib.pyplot as plt
import copy


class Measure:
    def __init__(self, locations: torch.tensor, weights: torch.tensor, device='cpu'):
        self.locations = torch.nn.parameter.Parameter(locations)
        self.weights = torch.nn.parameter.Parameter(weights)
        self.device = device
        self.grad_diff = 0.0

    def __str__(self) -> str:
        """
        Returns the locations and weights of the measure as a string.
        :returns: str
        """
        out = "\033[4mLocations:\033[0m     \033[4mWeights:\033[0m \n"
        for i in range(len(self.weights)):
            out += f'{self.locations[i].item(): < 10.9f}     {self.weights[i].item(): < 10.9f}\n'
        return out

    def __repr__(self) -> str:
        """
        Returns the locations and weights of the measure as a string.
        :returns: str
        """
        return self.__str__()

    def is_probability(self, tol=1e-6):
        """
        Returns True if the measure is a probability measure.
        :param tol: Tolerance for summing to 1
        """
        if torch.any(self.weights < 0):
            return False
        if torch.abs(self.weights.sum() - 1) > tol:
            return False
        return True

    def total_mass(self) -> float:
        """
        Returns the sum of all weights in the measure: \sum_{i=1}^n w_i
        :returns: float
        """
        return sum(self.weights).item()

    def total_variation(self) -> float:
        """
        Returns the sum of the absolute value of all weights in the measure: \sum_{i=1}^n |w_i|
        :returns: float
        """
        return sum(abs(self.weights)).item()

    def support(self, tol=5e-3):
        """
        :param tol: proportion of total variation that can be un-accounted for by the support.
        :returns: all index where the weights are non-zero
        """
        sorted_idx = torch.argsort(self.weights.abs())
        accum_weight = torch.cumsum(self.weights[sorted_idx].abs(), dim=0)
        cutoff = tol * self.total_variation()
        return sorted_idx[cutoff < accum_weight]

    def positive_part(self):
        """
        Returns the positive part of the Lebesgue decomposition of the measure
        """
        return Measure(self.locations, torch.max(self.weights, torch.zeros(len(self.weights))))

    def negative_part(self):
        """
        Returns the negative part of the Lebesgue decomposition of the measure
        """
        return Measure(self.locations, -torch.min(self.weights, torch.zeros(len(self.weights))))

    def sample(self, size):
        """
        Returns a sample of numbers from the distribution given by the measure
        :param size: Number of elements to sample
        :returns: sample of random numbers based on measure
        """
        if torch.any(self.weights < 0):
            assert ValueError("You can't have negative weights in a probability measure!")

        sampling = torch.multinomial(self.weights, size, replacement=True)
        sample = torch.tensor([self.locations[element.item()] for element in sampling])
        return sample

    def zero_grad(self):
        self.weights.grad = torch.zeros(len(self.weights), device=self.device)

    def visualize(self):
        """
        Visualization of the weights
        """
        plt.bar(self.locations.tolist(), self.weights.tolist(), width=0.1)
        plt.axhline(y=0, c="grey", linewidth=0.5)
        plt.show()


class Optimizer:

    def __init__(self, measure: Measure, lr : float = 0.1):
        self.measure = measure
        self.lr = lr
        self.state = {'measure':self.measure, 'lr':self.lr}

    def put_mass(self, mass, location_index):
        """
        In current form, this method puts mass at a specified location, s.t. the location still
        has mass less at most 1 and returns how much mass is left to distribute.
        :param mass: Mass left to take
        :param location_index: Index of location to take mass from
        :returns: mass left to add to measure after adding at specified location
        """
        with torch.no_grad():
            self.measure.weights[location_index] += mass

    def take_mass(self, mass, location_index) -> float:
        """
        In current form, this method takes mass from a specified location, s.t. the location still
        has non-negative mass and returns how much mass is left to take.
        :param mass: Mass left to take
        :param location_index: Index of location to take mass from
        :returns: mass left to remove from measure after removing from specified location
        """
        with torch.no_grad():
            if mass > self.measure.weights[location_index].item():
                mass_removed = self.measure.weights[location_index].item()
                self.measure.weights[location_index] = 0
            else:
                self.measure.weights[location_index] -= mass
                mass_removed = mass
        return mass_removed
    
    def stop_criterion(self, tol_supp=1e-6, tol_const=1e-3):
        """
        Checks if the difference between the maximum and minimum gradient is
        within a certain range.
        :param tol_supp: lower bound for wieghts considered
        :param tol_const: stop value, when the maximum difference of gradients
        is smaller than this value the minimization should seize
        """
        return self.measure.weights.grad[self.measure.support(tol_supp)].max() - self.measure.weights.grad.min() < tol_const

    def step(self):
        """
        Steepest decent with fixed total mass
        """

        # Sort gradient
        grad_sorted = torch.argsort(self.measure.weights.grad)

        # Distribute positive mass
        mass_pos = self.lr
        self.put_mass(mass_pos, grad_sorted[0].item())

        # Distribute negative mass
        mass_neg = self.lr
        for i in torch.flip(grad_sorted, dims=[0]):
            mass_neg -= self.take_mass(mass_neg, i.item())
            if mass_neg <= 0:
                break
    
    def update_lr(self, fraction = 0.7):
        """
        Updates learning rate for the optimizer

        :param lr: learning rate
        """
        self.lr *= fraction

    def state_dict(self):
        """
        Updates the state dictionary for the optimizer
        """
        print("\n".join("\t{}: {}".format(k, v) for k, v in self.state.items()))
        return self.state_dict

    def load_state_dict(self, state_dict):
        """
        Overloads the current state dictionary

        :param state_dict: state dictionary to load
        """
        self.state = state_dict

    def lr_criterion(self, loss_fn, measure):
        """
        Checks if learning rate should be decreased

        :param loss_fn: loss function
        :param measure: measure to compare current measure to
        """
        return loss_fn(self.measure.weights) - loss_fn(measure.weights) < 0

    def minimize(self, loss_fn, max_epochs=10000,smallest_lr=1e-6, silent=False, tol_supp=1e-6, tol_const=1e-3,):
        for epoch in range(max_epochs):
            measure=copy.deepcopy(self.measure)
            self.measure.zero_grad()
            loss=loss_fn(self.measure.weights)
            loss.backward()
            self.step()
            if self.stop_criterion(tol_supp, tol_const):
                print(f'\nOptimum is attained. Value of the goal function is {self.val}')
                self.is_optim = True
                return
            '''
            if self.lr_criterion(loss_fn,measure)==False:
                self.measure=measure
                self.update_lr()
'''
            if epoch % 100 == 0 and silent==False:
                print(f'Epoch: {epoch:<10} Loss: {loss:<10.0f} LR: {self.lr}')   

            if self.lr < smallest_lr:
                print(f'The step size is too small: {self.lr: 0.8f}')
                return
            




def main():
    a = torch.tensor([-0.1, 0.1, 0.3, 0.1, 0.4])
    b = torch.tensor([1., 2., 3., 4., 5.])

    c = Measure(b, a)
    # print(c.negative_part())
    # print(c.put_mass(0.9, 1))
    print(c)
    test_sample()


def test_sample():
    a = torch.tensor([0.1, 0.1, 0.3, 0.1, 0.4])
    b = torch.tensor([1., 2., 3., 4., 5.])

    d = Measure(b, a)
    print(d)
    print(d.is_probability())
    d.visualize()

    print(d.sample(2000))

def test_state_dict():
    u = Measure(torch.tensor([0.,0.5,1.]),torch.tensor([0.2,0.5,0.3]))
    opt = Optimizer(u)
    opt.state_dict()

if __name__ == "__main__":
    main()
