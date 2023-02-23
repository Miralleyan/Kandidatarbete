import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class TorchMeasure(nn.Module):
    def __init__(self, locations, weights, optimize_locations = False):
        super().__init__()
        assert len(locations) == len(weights), 'locations and weights should have the same length!'
        if optimize_locations:
            self.locations = nn.parameter.Parameter(torch.tensor(locations, dtype= torch.float))
            # self.locations = torch.tensor(locations, dtype=torch.float64, requires_grad=True)
        else:
            self.locations = torch.tensor(locations, dtype= torch.float)

        # self.weights = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
        self.weights = nn.parameter.Parameter(torch.tensor(weights, dtype= torch.float))
        self.n = len(weights)


    def __str__(self): # prints with print command
        '''
          Returns the locations and weights of the measure as a string. Ignore atoms of
          the absolute value of weight smaller than tol_weight
          :returns: str
          '''
        out = "\033[4mLocations:\033[0m \033[4mWeights:\033[0m \n"
        for i in range(len(self.weights)):
            out += f'{self.locations[i].item()}           {self.weights[i].item(): < 6.6f}\n'
        return out

    def __repr__(self):
        '''
         Prints on carriage return the locations and weights of the measure as a string.
         :returns: str
         '''
        return self.__str__()

    def atoms(self, tol_weight=0.):
        '''
          Returns the locations and weights of the atoms. Ignores atoms of
          the absolute value of weight smaller than tol_weight
          :param: tol_weight - tolerance of a weight to be considered as 0. By default, all
          atoms with positive weight are returned
          :returns: str
          '''
        a = []
        for i in range(len(self.weights)):
            if abs(np.abs(self.weights[i].item()) > tol_weight):
                a.append([self.locations[i].item(), self.weights[i].item()])
        return a

    def total_variation(self):
        '''
        :return: the total variation of the measure
        '''
        return float(self.weights.abs().sum())

    def total_mass(self):
        '''
        :return: the total mass of the measure
        '''
        return float(self.weights.sum())

    # def support(self, tol_supp=1e-12):
    #     """ Support of the measure up to tol_supp tolerance of what is considered as 0 """
    #     tol = self.total_variation() * tol_supp
    #     return np.arange(self.n)[torch.abs(self.weights) > tol]

    def support(self, tol_prop = 1e-2):
        '''
        :param tol_prop: proportion of total variation that can be un-accounted for by the support.
        :returns: all indices of the atoms accounting for at least total.variation * (1-tol) mass
        '''
        sorted_idx = torch.argsort(self.weights.abs())
        accum_weight = torch.cumsum(self.weights.abs()[sorted_idx], dim=0)
        cutoff = tol_prop * self.total_variation()
        supp = sorted_idx[cutoff < accum_weight]
        return supp[supp.argsort()]

    def is_positive(self, tol_pos=1e-9):
        '''
        Check if the measure is non-negative up to tolerance for small negative atoms
        :param tol_pos: tolerance of a weight to be considered non-negative
        :return: Boolean
        '''
        tol = self.total_variation() * tol_pos
        return bool(torch.all(self.weights >= -tol))

    def is_probability(self, tol=1e-6):
        '''
        Check whether the measure is probability up to a tolerance
        :param tol: tolerance of the total mass to be 1
        :return: Boolean
        '''
        return self.is_positive() and np.abs(self.total_mass() - 1) <= tol

    def copy(self):
        '''
        :return: a copy of the measure'''
        return TorchMeasure(self.locations.detach().numpy(), self.weights.detach().numpy())

    # def sample_from(self, nmb):
    #     """ sample n locations from probability distribution proportional to the measure """
    #     assert self.is_positive(), 'The measure should be positive to sample from it!'
    #     cdf = np.cumsum(self.weights.detach().numpy())/self.total_mass()
    #     sample = []
    #     for i in range(nmb):
    #         sample.append(self.locations.detach().numpy()[cdf > np.random.random()][0]) # inverse of the cdf method
    #     return np.array(sample)

    def sample(self, size):
        '''
        Returns a sample of numbers from the distribution given by the measure
        :param size: Number of elements to sample
        :returns: sample of random numbers distributed as the measure scaled to have mass 1
        '''
        assert self.is_positive(), 'The measure should be positive to sample from it!'

        sampling = torch.multinomial(self.weights, size, replacement = True)
        sample = torch.tensor([self.locations[element.item()] for element in sampling])
        return sample

    def visualize(self):
        '''
        Visualization of the weights
        '''
        plt.bar(self.locations.tolist(), self.weights.tolist(), width=0.1)
        plt.axhline(y=0, c="grey", linewidth=0.5)
        plt.show()

    def put_mass(self, grad: torch.Tensor, epsilon: float):
        """ Put mass epsilon to the coordinate of weights where grad is minimal """
        with torch.no_grad():
            self.weights[grad.argmin()] += epsilon
        ## without "with" above autograd gets confused when tensor with gradient is changed in place
        # see https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf


    def take_mass(self, grad: torch.Tensor, epsilon: float):
        '''
        Take mass epsilon from the coordinates of weights where grad is maximal starting from the maximal,
        next to maximal, etc.
        :param grad: gradient
        :param epsilon: step size - half the total variation of the increment
        :return: TorchMeasure
        '''
        assert epsilon < 0.5 * self.total_variation() , 'take_mass(): step size is too big!'
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
        '''
        Making the true gradient descent step preserving the total mass of the measure
        :param grad: gradient
        :param epsilon: step size
        :param silent: if True, print the resultant measure
        :return: TorchMeasure
        '''
        """ """
        self.put_mass(grad, epsilon)
        self.take_mass(grad, epsilon)
        if not silent:
            return print(self)

    def stop_criterion(self, grad, tol_supp=1e-3, tol_const=1e-3, adaptive=False):
        '''
        The first-order necessary condition for minimum of a functional of measures:
        the gradient should be minimal and constant on the support of the measure.
        :param grad: gradient
        :param tol_supp: tolerance of the support - ignore atoms with weight smaller than
        :param tol_const: tolerance of a constant value - consider gradient as a constant
        if it varies by less than tol_const on the support
        :param adaptive: if True, consider gradient a constant if it varies by less
        than tol_const * (range of grad) on the support. Useful when to set the tolerance
        scale by taking account of the scale of the gradient. But may lead to the criterion
        not to be satisfied.
        :return: Boolean
        '''
        min_grad, max_grad = grad.min(), grad.max()
        if adaptive:
            tol_const *= (max_grad - min_grad)
        return bool(grad[self.support(tol_supp)].max() - min_grad < tol_const)
# ____________________________________________________________________________________________________________________

class MeasureMinimizer(nn.Module):
    ''' : param
        mes - initial measure of class TorchMeasure
        goal_fn - the goal function of measure to be minimised
        lr - initial learning rate.
        Computed values:
        mes - current measure
        val - current value of the goal function
        grad - current value of the gradient of goal_fn at mes
        is_optim - boolean: if mes satisfies the necessary criterion for the minimum
        '''

    def __init__(self, mes: TorchMeasure, goal_fn, learning_rate = 0.1, **goal_func_kwargs):
        super(MeasureMinimizer, self).__init__()
        self.mes = mes
        self.goal_fn = goal_fn
        self.lr = learning_rate
        self.goal_func_kwargs = goal_func_kwargs
        # self.grad = None
        # computed initial values:
        self.val = self.goal_fn(self.mes, **self.goal_func_kwargs)
        self.mes.weights.grad = None # clear the gradients
        self.val.backward() # compute the gradient
        self.grad = self.mes.weights.grad
        self.is_optim = self.mes.stop_criterion(self.grad)

    def step(self, lr = None, armijo_val = 0.7):
        ''' Try step with learning rate lr. If the goal function is reduced, update the measure,
        value of the function and the gradient
        Otherwise, reduce step size and return '''
        if lr is None:
            lr=self.lr
        val_is_reduced = False
        mes_new = self.mes.copy()
        mes_new.take_step(self.grad, lr)
        val = self.goal_fn(mes_new, **self.goal_func_kwargs)
        if val < self.val: # goal function is reduced, update mes
            val_is_reduced = True
            self.val = val
            val.backward()
            self.grad = mes_new.weights.grad
            self.mes = mes_new
            self.lr = lr
        else:
            if val > self.val:
                lr = lr * armijo_val  # reduce the learning rate
            else: # value is the same, still move to new measure
                print('The value of the goal function has not changed!')
                val.backward()
                self.grad = mes_new.weights.grad
                self.mes = mes_new

        return lr, val_is_reduced

    def minimize(self, max_no_steps=1000, armijo_val = 0.7, smallest_lr = 1e-6,
                 silent = False, print_each_step=10,
                 tol_supp=1e-6, tol_const=1e-3, adaptive=False):
        '''Minimize goal_fn of a measure starting from initial measure mes
             using the true steepest descent in the space of measures with a fixed mass starting
             with initial learning rate lr. If the goal_fn does not optimize at the current lr,
             diminish its size armijo_val times.
             Stop either when the necessary optimality criterion is satisfied, the max_no_steps reached
              or the learning rate becomes smaller than smallest_lr.
              Print details of optimization if silent=False each 10th (print_each_step) step'''

        for k in range(max_no_steps):
            if self.mes.stop_criterion(self.grad, tol_supp, tol_const, adaptive):
                print(f'\nOptimum is attained. Value of the goal function is {self.val}')
                self.is_optim = True
                return
            if not silent and k % print_each_step == print_each_step - 1:
                print(f'Step {k + 1}: Goal function={self.val}, Learning rate={self.lr: 0.8f}')
            if silent:
                if k % 10 == 9:
                    print('.')
                else:
                    print('.', end="")
            self.lr, val_reduced = self.step(armijo_val=armijo_val)
            if not val_reduced: # value of the function is not smaller so the learning rate is decreased
                if (self.lr < smallest_lr):
                    print(f'The step size is too small: {self.lr: 0.8f}')
                    return

