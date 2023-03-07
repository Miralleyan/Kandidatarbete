import torch
import matplotlib.pyplot as plt
import copy


class Measure:
    def __init__(self, locations: torch.tensor, weights: torch.tensor, device='cpu', optim_locations = False):
        self.locations = locations
        self.weights = torch.nn.parameter.Parameter(weights)
        self.device = device
        self.grad_diff = 0.0

    def __str__(self) -> str:
        """
        Returns the locations and weights of the measure as a string.
        :returns: str
        """
        out = "\033[4mLocations:\033[0m".ljust(28) + "\033[4mWeights:\033[0m \n"
        for i in range(len(self.weights)):
            out += f'{self.locations[i].item():<20.9f}{self.weights[i].item():<.9f}\n'
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
        Returns a sample of indeces from the locations of the measure 
        given by the distribution of the measures weights
        :param size: Numreturn torch.dot(errors, measures[0].weights) + torch.dot(errors, measures[1].weights)ber of elements to sample
        :returns: sample of indeces for random numbers based on measure
        """
        if torch.any(self.weights < 0):
            assert ValueError("You can't have negative weights in a probability measure!")

        sample_idx = torch.multinomial(self.weights, size, replacement=True)
        sample = self.locations[sample_idx]
        return sample

    def zero_grad(self):
        self.weights.grad = torch.zeros(len(self.weights), device=self.device)

    def visualize(self):
        """
        Visualization of the weights
        """
        plt.bar(self.locations.tolist(), self.weights.tolist(), width=0.1)
        plt.axhline(y=0, c="grey", linewidth=0.5)
        plt.draw()


class Optimizer:

    def __init__(self, measures, lr : float = 0.1):
        if type(measures) == Measure:
            self.measures = [measures]
        elif type(measures) != list:
            Exception('Error: measures has to be of type Measure or list')
        else:
            self.measures = measures
        self.lr = [lr]*len(self.measures)
        self.old_lr = [lr]*len(self.measures)
        self.state = {'measure':self.measures, 'lr':self.lr}
        self.is_optim = False

    def put_mass(self, meas_index, mass, location_index):
        """
        In current form, this method puts mass at a specified location, s.t. the location still
        has mass less at most 1 and returns how much mass is left to distribute.
        :param mass: Mass left to take
        :param location_index: Index of location to take mass from
        :returns: mass left to add to measure after adding at specified location
        """
        with torch.no_grad():
            self.measures[meas_index].weights[location_index] += mass

    def take_mass(self, meas_index, mass, location_index) -> float:
        """
        In current form, this method takes mass from a specified location, s.t. the location still
        has non-negative mass and returns how much mass is left to take.
        :param mass: Mass left to take
        :param location_index: Index of location to take mass from
        :returns: mass left to remove from measure after removing from specified location
        """
        with torch.no_grad():
            if mass > self.measures[meas_index].weights[location_index].item():
                mass_removed = self.measures[meas_index].weights[location_index].item()
                self.measures[meas_index].weights[location_index] = 0
            else:
                self.measures[meas_index].weights[location_index] -= mass
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
        return min([measure.weights.grad[measure.support(tol_supp)].max()
                    - measure.weights.grad.min() < tol_const for measure in self.measures])

    def step(self, meas_index):
        """
        Steepest decent with fixed total mass
        """

        # Sort gradient
        grad_sorted = torch.argsort(self.measures[meas_index].weights.grad)

        # Distribute positive mass
        mass_pos = self.lr[meas_index]
        self.put_mass(meas_index, mass_pos, grad_sorted[0].item())

        # Distribute negative mass
        mass_neg = self.lr[meas_index]
        for i in torch.flip(grad_sorted, dims=[0]):
            mass_neg -= self.take_mass(meas_index, mass_neg, i.item())
            if mass_neg <= 0:
                break
    
    def update_lr(self, fraction = 0.7):
        """
        Updates learning rate for the optimizer

        :param lr: learning rate
        """
        self.lr = [lr*fraction for lr in self.lr]

    def state_dict(self):
        """
        Updates the state dictionary for the optimizer
        """
        print("\n".join("\t{}: {}".format(k, v) for k, v in self.state.items()))
        return self.state

    def load_state_dict(self, state_dict):
        """
        Overloads the current state dictionary
\chi ^{2} = Pearson's cumulative test statistic, which asymptotically approaches a χ 2 \chi ^{2} distribution.
        :param state_dict: state dictionary to load
        """
        self.state = state_dict

    def lr_decrease_criterion(self, loss_fn, measure, old_measure):
        """
        Checks if learning rate should be decreased

        :param loss_fn: loss function
        :param measure: measure to compare current measure to
        """
        return loss_fn(old_measure) - loss_fn(measure) < 1e-3

    def minimize(self, loss_fn, max_epochs=10000,smallest_lr=1e-6, tol_supp=1e-6, tol_const=1e-3, verbose=False, print_freq=100):
        #Suceeded=True
        old_loss = float('inf')
        for epoch in range(max_epochs):
            #if Suceeded==True:
                #self.lr=[lr for lr in self.old_lr]
            old_measures = copy.deepcopy(self.measures)
            for m in self.measures:
                m.zero_grad()
            loss = loss_fn(self.measures)
            loss.backward()
            for meas_index in range(len(self.measures)):
                self.step(meas_index)

            if self.lr_decrease_criterion(loss_fn, self.measures, old_measures):
                self.measures = old_measures
                self.update_lr()
            elif self.stop_criterion(tol_supp, tol_const):
                print(f'\nOptimum is attained. Loss: {loss}. Epochs: {epoch} epochs.')
                self.is_optim = True
                return
            '''
            if old_loss < loss:
                #Suceeded=False
                self.measures = old_measures
                self.update_lr()
            #else:
            #    Suceeded=True
            '''
            if verbose and epoch % print_freq == 0:
                print(f'Epoch: {epoch:<10} Loss: {loss:<10.0f} LR: {self.lr}')

            if min([lr < smallest_lr for lr in self.lr]):
                print(f'The step size is too small: {min(self.lr)}')
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
