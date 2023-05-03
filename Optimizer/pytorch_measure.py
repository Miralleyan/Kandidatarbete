import torch
import matplotlib.pyplot as plt
import copy
import scipy 
import itertools
import numpy as np
import time


class Measure:
    def __init__(self, locations: torch.tensor, weights: torch.tensor, device='cpu', optim_locations = False):
        self.locations = locations
        self.weights = torch.nn.parameter.Parameter(weights)
        self.device = device

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
        :param size: Number of elements to sample
        :returns: sample of indeces for random numbers based on measure
        """
        if torch.any(self.weights < 0):
            assert ValueError("You can't have negative weights in a probability measure!")

        sample_idx = torch.multinomial(self.weights, size, replacement=True)
        sample = self.locations[sample_idx]
        return sample

    def copy(self):
        return Measure(self.locations, self.weights)

    def zero_grad(self):
        self.weights.grad = None #torch.zeros(len(self.weights), device=self.device)

    def visualize(self):
        """
        Visualization of the weights
        """
        plt.bar(self.locations.detach(), self.weights.detach(), width=0.1, label="Measure")
        plt.axhline(y=0, c="grey", linewidth=0.5)
        plt.draw()


class Optimizer:

    def __init__(self, measures, loss : str, lr : float = 0.1):
        # Create list of measures
        if type(measures) == Measure:
            self.measures = [measures]
        elif type(measures) != list:
            Exception('Error: measures has to be of type Measure or list')
        else:
            self.measures = measures
        # Create list of lr's
        if type(lr) == float or type(lr) == int:
            self.lr = [lr]*len(self.measures)
        elif type(lr) != list:
            Exception('Error: lr has to be of type float or list')
        else:
            self.lr = lr

        loss_dict = {'essr':self.essr, 'nll':self.nll, 'KDEnll':self.KDEnll, 'chi_squared':self.chi_squared}
        self.loss = loss_dict[loss]
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
                self.measures[meas_index].weights[location_index] = 0.
            else:
                self.measures[meas_index].weights[location_index] -= mass
                mass_removed = mass
        return mass_removed
    
    def stop_criterion(self, tol_supp=1e-6, tol_const=1e-2, adaptive = False):
        """
        Checks if the difference between the maximum and minimum gradient is
        within a certain range.
        :param tol_supp: lower bound for wieghts considered
        :param tol_const: stop value, when the maximum difference of gradients
        is smaller than this value the minimization should seize
        """
        if adaptive:
            return min([measure.weights.grad[measure.support(tol_supp)].max()
                        - measure.weights.grad.min() < tol_const*(measure.weights.grad.max() - measure.weights.grad.min())
                        for measure in self.measures])
        else:
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
            if mass_neg <= 0.:
                break
    
    def update_lr(self, index, fraction=0.7):
        """
        Updates learning rate for the optimizer

        :param fraction: multiply lr with this value
        """
        self.lr[index]*=fraction

    def state_dict(self):
        """
        Updates the state dictionary for the optimizer
        """
        print("\n".join("\t{}: {}".format(k, v) for k, v in self.state.items()))
        return self.state

    def load_state_dict(self, state_dict):
        """
        Overloads the current state dictionary

        :param state_dict: state dictionary to load
        """
        self.state = state_dict

    def lr_decrease_criterion(self, loss_fn, measure, old_measure):
        """
        Checks if learning rate should be decreased

        :param loss_fn: loss function
        :param measure: measure to compare current measure to
        """
        return loss_fn(old_measure) < loss_fn(measure)

    def minimize(self, data, model, h = 0, alpha = 0.001, max_epochs=2000, smallest_lr=1e-6, verbose=False,
                 tol_supp=1e-6, tol_const=1e-2,  print_freq=100, adaptive=False,test=False):
        """
        :param data: list of tensors of data points. If x and y, then data should be on the form [x_tensor,y_tensor].
        If only one series of data points, just this tensor is needed
        :param model: model written as a function.
        Should take a data input (x) and a set of parameters (params).
        :param max_epochs: Max number of iterations
        :param smallest_lr: Minimizer wil stop when lr is below this value
        :param verbose: Print information about each epoch
        :param print_freq: How frequently the minimizer should print information
        :param tol_supp:
        :param tol_const:
        :param adaptive:
        :return: Optimized measures
        """
        lr = self.lr
        loss_fn = self.loss

        if type(data) == torch.tensor:
            data = [data, data]

        perms, prep = self.prep_step(data, model, h, alpha)
        
        if test:
            tid=[]
            LossNotChanged=0
            t1=time.time()

        for epoch in range(max_epochs):
            # Backup current measures and reset grad
            old_measures = copy.deepcopy(self.measures)
            for m in self.measures:
                m.zero_grad()
            

            # Compute loss and grad
            loss_old = loss_fn(perms, *prep)
            loss_old.backward()

            # Stop criterion
            if self.stop_criterion(tol_supp, tol_const, adaptive):
                print(f'\nOptimum is attained. Loss: {loss_old}. Epochs: {epoch} epochs.')
                self.is_optim = True
                if test:
                    t2=time.time()
                    return self.measures,t2-t1,epoch
                else: 
                    return self.measures
            
            if min(lr) < smallest_lr:
                print(f'The step size is too small: {lr}')
                if test:
                    t2=time.time()
                    if LossNotChanged<5:
                        return self.measures,t2-t1,epoch
                    else:
                        return self.measures, tid[0][0],tid[0][1]
                else: 
                    return self.measures

            # Step
            maxima = []
            for meas_index in range(len(self.measures)):
                sup_index = self.measures[meas_index].support()
                grads = copy.deepcopy(self.measures[meas_index].weights.grad)
                #print(grads)
                maxima.append(torch.max(grads[sup_index]))
            max_index = maxima.index(sorted(maxima)[-1])
            self.step(max_index)

            loss_new = loss_fn(perms, *prep)
            loss_new.backward()

            # bad step
            if loss_old < loss_new:
                # Revert to the backup measure and decrease lr
                self.measures = copy.deepcopy(old_measures)
                self.update_lr(max_index, fraction=0.1)

                if verbose:
                    print(f'Epoch: {epoch:<10} Lr was reduced to: {lr}')
            elif loss_old == loss_new and verbose:
                if test and LossNotChanged<5:
                    LossNotChanged+=1
                    t2=time.time()
                    tid.append([t2-t1,epoch])
                print(f'Epoch: {epoch:<10} Loss did not change ({loss_new})')

            # successful step
            else:
                if test and LossNotChanged<5:
                    LossNotChanged=0
                    tid=[]
                if epoch % print_freq == 0:
                    if verbose:
                        print(f'Epoch: {epoch:<10} Loss: {loss_new:<10.9f} LR: {lr}')
                    else:
                        print('.')

        
        print('Max epochs reached')
        if test:
            t2=time.time()
            if LossNotChanged<5:
                return self.measures,t2-t1,epoch
            else:
                return self.measures, tid[0][0],tid[0][1]
        else: 
            return self.measures

    def visualize(self):
        """
        Visualizes the measures of the optimizer. Currently requires that
        a gradient has been stored in the weights of the measures.
        """
        cols = int(torch.ceil(torch.sqrt(torch.tensor(len(self.measures)))).item())
        rows = int(torch.ceil(torch.tensor(len(self.measures)/cols)).item())
        fig, axs = plt.subplots(rows,cols)
        axs = axs.flatten()
        fig.suptitle('Optimizer Visualization')
        grads = [measure.weights.grad for measure in self.measures]
        for i, measure in enumerate(self.measures):
            M, m = grads[i].max(), grads[i].min()
            wm = measure.weights.max()
            scaled_weights = measure.weights * (M - m) / wm * 0.25 + m
            support = measure.support()
            with torch.no_grad():
                # Support locations
                axs[i%cols+(i//cols)*cols].plot(measure.locations[support], torch.zeros(len(measure.weights))[support] + m,
                            '.', c='red', label=' Measure Support')
                # Gradient
                axs[i%cols+(i//cols)*cols].plot(measure.locations, grads[i], c='green', label=' Gradient')
                # Measure weights where there is support
                axs[i%cols+(i//cols)*cols].vlines(measure.locations[support], torch.zeros(len(measure.weights))[support] + 1.3 * m,
                              scaled_weights[support], colors='blue', label=' Measure Weights')
                axs[i%cols+(i//cols)*cols].axhline(y=m, c="orange", linewidth=0.5)
                axs[i%cols+(i//cols)*cols].legend(loc='upper right')
                axs[i%cols+(i//cols)*cols].set_ylim([m, M])

    # Loss functions
    def essr(self, perms, errors):
        """
        Calculates the expected sum of square residuals loss function

        :param perms: list of the possible permutations of one location from each measure
        :param errors: tensor of the sum of errors, compared with true data, for each permutation of locations
        """
        probs = torch.cat([self.measures[i].weights[perms[:, i]].unsqueeze(1) for i in range(len(self.measures))], 1).prod(1)
        return errors.dot(probs)
    
    def nll(self, perms, loc_index):
        """
        Calculates the negative log-likelihood loss function

        :param perms: list of the possible permutations of one location from each measure
        :param loc_index: list of location permutation closest with the least absolute error for each data point
        """
        probs = torch.cat([self.measures[i].weights[perms[:, i]].unsqueeze(1) for i in range(len(self.measures))], 1).prod(1)
        return -sum(torch.log(probs[loc_index]))
    
    def KDEnll(self, perms, kde_mat, h):
        """
        Calculates the negative log-likelihood loss function, with a KDE

        :param perms: list of the possible permutations of one location from each measure
        :param kde_mat: matrix of kernels for each location permutation
        :param h: bandwidth for the KDE
        """
        probs = torch.cat([self.measures[i].weights[perms[:, i]].unsqueeze(1) for i in range(len(self.measures))], 1).prod(1)
        return -(torch.matmul(kde_mat, probs) / h).log().sum()
    
    def chi_squared(self, perms, bins_freq):
        """
        Calculates the chi-squared loss function

        :param perms: list of the possible permutations of one location from each measure
        :param bins_freq: frequencies of data points closest to each location permutation
        """
        probs = torch.cat([self.measures[i].weights[perms[:, i]].unsqueeze(1) for i in range(len(self.measures))], 1).prod(1)
        return (probs**2/bins_freq).sum()
    
    # Preparational step for loss functions
    def prep_step(self, data, model, h = 0, alpha = 0.001):
        """
        Does a preparatory step and returns the prepared data needed for the optimizers loss function

        :param data: data to fit the model to
        :param model: model that should be fit to data, should be a function taking the data (x) and locations as parameters
        :param h: bandwidth parameter for KDE
        :param aplha: label smoothing parameter for chi-squared loss function
        """
        perms = torch.tensor([item for item in itertools.product(*[range(measure.weights.size(dim=0)) for measure in self.measures])])
        locs = torch.cat([self.measures[i].locations[perms[:, i]].unsqueeze(1) for i in range(len(self.measures))], 1)
        prep = []
        if self.loss == self.essr:
            prep.append(torch.tensor([(model(data[0], locs[i]) - data[1]).pow(2).sum() for i in range(len(perms))])) # errors
        elif self.loss == self.nll:
            loc_idx = []
            for i in range(len(data[0])):
                ab = torch.abs(model(data[0][i], [locs[:,i] for i in range(locs.size(dim=1))]) - data[1][i])
                loc_idx.append(torch.argmin(torch.tensor(ab)))
            prep.append(torch.tensor(loc_idx))
        elif self.loss == self.KDEnll:
            if h == 0:
                h = 1.06*len(data[0])**(-1/5)
            sigma=torch.std(data[0])
            A=min(sigma,(torch.quantile(data[0],0.75)-torch.quantile(data[0],0.25))/1.35)
            h=0.9*A*len(data[0])**(-1/5)
            kde_mat = 1/np.sqrt(2*np.pi)*np.exp(-((data[1].view(-1,1) - model(data[0].view(-1,1), locs.transpose(0,1))) / h)**2/2)
            prep.append(kde_mat)
            prep.append(h)
        elif self.loss == self.chi_squared:
            loc_idx = []
            for i in range(len(data[0])):
                ab = torch.abs(model(data[0][i], [locs[:,i] for i in range(locs.size(dim=1))]) - data[1][i])
                loc_idx.append(torch.argmin(ab))
            bins = torch.tensor(loc_idx)
            bins_freq = torch.bincount(bins, minlength=np.cumprod([self.measures[i].weights.size(dim=0) for i in range(len(self.measures))])[-1])/len(data[0])**2
            bins_freq = bins_freq*(1-alpha)+alpha / len(bins_freq)
            prep.append(bins_freq)
        return perms, prep



class Check():
    def __init__(self, opt: Optimizer, model, x: torch.tensor,y: torch.tensor, alpha=0.05,normal=False,Return=False):
        """
        A class that will check how close a fitted measure is to the orginal by creating a confidence intervall att each x-value and checking to see if the corresponding
        y-value is insiede the confidence interval.

        :param opt: An instance of the class Optimizer
        :param model: A function with input: a list of x-values and a list of lists containing weights from measures
        :param x: The x-values used in the model
        :param y: The y-values from the original distribution that we use to fitt the measure
        :param alpha: The amount of confidence we want for our donfidence intervals, standard is 0.05 which corresponds to a 95% CI
        :param normal: Set to true if you know that the distribution is normal
        :param Return: Set to true if you wish the class to return the amount of misses and the bounds for the 95% CI
        """
        self.opt=opt
        self.model=model
        self.data=[x,y]
        self.N=len(x)
        self.alpha=alpha
        self.normal=normal
        self.Return=Return


    def check(self):
        '''
        Calculates the amount of the original data (y) that is outside
        the boundaries of a 95% confidence intervall (if no value is given to
        the variable alpha) and then calculates the probability of
        that amount of misses
        
        '''
        bounds=[]
        for x in self.data[0]:
            input=[]
            for meas in self.opt.measures:
                input.append(meas.sample(self.N))
            bounds.append(self.CI(self.model(x,input)))
        miss=self.misses(self.data[1],bounds)

        lci=scipy.stats.binom.ppf(self.alpha/2,self.N,self.alpha)
        hci=scipy.stats.binom.ppf(1-self.alpha/2,self.N,self.alpha)
        if (miss >= lci) and (miss <= hci):
            print(f'{miss} is inside the confidence interval ({lci}, {hci}):')
            print(f'No contradiction with the fitted model at {100*(1-self.alpha)}% confidence level')
        else:
            print(f'{miss} is outside the confidence interval ({lci}, {hci}):')
            print(f'Number of misses is significantly at {100*(1-self.alpha)}% confidence level different from expected for the fitted model!')
        if self.Return==True:
            return lci,hci, miss
        
    def CI(self, data:list[float]):
        '''
        Calculates the bounds of an approximate 95% confidence intervall
        for the given data in output
        :param data: List of values that the confidence interval is calculated from 
        '''
        if self.normal:
            mean=torch.mean(data)
            std=torch.std(data)
            q=scipy.stats.norm.ppf(self.alpha/2)
            cil=mean+q*std
            cih=mean-q*std
            bounds=[cil,cih]
        else:
            edge=int(self.alpha/2*self.N)
            idx_sorted_cropped=torch.argsort(data)[edge:self.N-edge]
            bounds=data[idx_sorted_cropped[[0,-1]]]
        return bounds
    

    def misses(self,y:list[float],bounds:list[list[float,float]]):
        '''
        Calculates the amount of values in y that are not within the 
        corresponding boundary in bounds

        :param y: The data which we will se if it is in the corresponding confidence interval
        :param bounds: list containing the pairs of bounds for each x
        '''
        miss=0
        for i in range(len(y)):
            if y[i]>bounds[i][1] or y[i]<bounds[i][0]:
                miss+=1
        return miss