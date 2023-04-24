## Gaussian Regression model:
## eta(x) = sum_k beta_k h_k(x)
## where h_k(x) are given functions and
## beta_k are normally distributed independent random variables
## with mean m_k and variance v_k
## eta(x) is then also normally distributed with
## the mean m(x) = sum_k m_k h_k(x)  and
## the variance v(x) = sum_k v_k h_k(x)**2

## Given the data (x_i, y_i), the parameters m_k, v_k are estimated
## by maximising the log-likelihood
## sum_i log f_{eta(x_i)} (y_i)
## where f_{eta(x)} is the density of the Normal N(m(x), v(x)) distribution


import numpy as np
import torch

import matplotlib.pyplot as plt
from scipy.stats import norm, binom
# from torch.distributions.normal import Normal
# log_prob is the log pdf


class GaussianRegression:
    def __init__(self, funcs: list, means=None, st_devs=None):
        '''
        :param funcs: list of torch functions
        :param means: means of the coefficients
        :param st_devs: standard deviations of the coefficients
        '''
        self.funcs = funcs
        self.dim_regression = len(funcs)
        if means is None:
            self.means = torch.zeros(self.dim_regression, requires_grad=True)
        else:
            self.means = torch.tensor([float(m) for m in means], requires_grad=True)
        if st_devs is None:
            self.st_devs = torch.ones(self.dim_regression, requires_grad=True)
        else:
            self.st_devs = torch.tensor([float(s) for s in st_devs], requires_grad=True)

    def average(self, X, means=None):
        '''
        :param X: input locations
        :param means: the means of the coefs, if none, self.means are used
        :return: expected value of the regression at each location from X
        '''
        if means is None:
            means = self.means
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        ave = torch.tensor(())
        for x in X:
            vals = torch.tensor([0.])
            for j in range(self.dim_regression):
                vals += means[j] * self.funcs[j](x)
            ave = torch.cat((ave, vals),0)
        return ave

    def deviation(self, X, st_devs=None):
        '''
        :param X: input locations
        :param st_devs: the st.dev's of the coefs, if none, self.st_devs are used
        :return: standard deviations of the regression at each location from X
        '''
        if st_devs is None:
            st_devs = self.st_devs
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        std = torch.tensor(())
        for x in X:
            var = torch.tensor([0.])
            for j in range(self.dim_regression):
                var += (st_devs[j] * self.funcs[j](x))**2
            std = torch.cat((std, torch.sqrt(var)), 0)
        return std

    def simulate(self, X, means=None, st_devs=None):
        '''
        simulate regression
        :param X: input locations
        :param means: the means of the coefs, if none, self.means are used
        :param st_devs: the st.dev's of the coefs, if none, self.st_devs are used
        :return: values of the regression at each location from X
        '''
        if means is None:
            means = self.means
        if st_devs is None:
            st_devs = self.st_devs
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        sim = torch.tensor(())
        for x in X:
            s = torch.tensor([0.])
            for j in range(self.dim_regression):
                s += (torch.randn(1) * st_devs[j] + means[j]) * self.funcs[j](x)
            sim = torch.cat((sim, s), 0)
        return sim

    def minus_log_pr(self, y, mean: torch.Tensor=torch.tensor([0.]), std: torch.Tensor=torch.tensor([1.])):
        return torch.log(std) + 0.5*((y - mean) / std)**2 # ignoring constant log(2*pi)

    def loglik(self, X: torch.Tensor,Y: torch.Tensor):
        log_probs = torch.tensor([0.])
        for i in range(len(X)):
            log_probs += self.minus_log_pr(Y[i], self.average([X[i]]), self.deviation([X[i]]))
        return(log_probs)

    def coefs_classic(self, x, y):
        ''' Fit the coefficients beta of the standard linear regression:
        y_i = sum_j beta_j h_j(x_i)+ epsilon_i, or Y = X beta + epsilon.
        The estimate of beta is
        beta_hat = (X' X)^(-1) X' y
        where X=||h_j(x_i)|| and X' is transpose of X
        :param x: column tensor of the input
        :param y: column tensor of the response
        :return: row tensor of the estimated beta_hat
        '''
        X = self.funcs[0](x.reshape(-1, 1))
        for f in self.funcs[1:]:
            X = torch.cat((X, f(x.reshape(-1, 1))), dim=1)
        XX_inv = torch.inverse(torch.matmul(X.transpose(0,1), X))
        coefs = torch.matmul(torch.matmul(XX_inv, X.transpose(0,1)), y.reshape(-1, 1))
        return torch.tensor([c for c in coefs], requires_grad=True)

    def fit(self, data_x, data_y, params=None, start_means=None, start_st_devs=None,
            algo=torch.optim.SGD, no_steps=1000, lr=0.01,verbose=False):
        '''
        Estimate the parameters means and/or st-devs by maximizing the log-likelihood
        :param data_x:  locations data
        :param data_y: response data
        :param params: if None, fit both means and st_devs,
            if params=[self.means] - fit only means,
            if params=[self.st_devs] - fit only standard deviations
        :param start_means: starting values for the means. If none, estimate from the
            ordinary (non-stochastic) regression is used.
        :param start_st_devs: starting values for the st_devs. If None, self.st_devs is used
        :param algo: optimizer algorithm to be used, e.g. torch.optim.Adam
        :param no_steps: the number of steps for minimization
        :param lr: learning rate
        :param : if True, print the current values of minus log-likelihood and lr
        :return: updated self.means and/or self.st_devs and the value of minus log-likelihood
        '''
        if type(data_x) is not torch.Tensor:
            data_x = torch.tensor(data_x)
        if type(data_y) is not torch.Tensor:
            data_y = torch.tensor(data_y)

        if params is None:
            params = [self.means, self.st_devs] # estimate both means and st_devs

        # As the starting point for the means, if not given,
        # take the coefficients of the standard linear regression: Y = X beta + epsilon
        if start_means is None:
            self.means = self.coefs_classic(data_x, data_y)
        else:
            if type(start_means) is not torch.Tensor:
                self.means = torch.tensor(start_means)
            else:
                self.means = start_means

        if start_st_devs is not None:
            if type(start_st_devs) is not torch.Tensor:
                self.st_devs = torch.tensor(start_st_devs)
            else:
                self.st_devs = start_st_devs

        ## Turning on the grads
        self.means.requires_grad_(True)
        self.st_devs.requires_grad_(True)

        optimizer = algo(params, lr=lr)

        for steps in range(no_steps):
            LL = self.loglik(data_x, data_y)
            optimizer.zero_grad()
            LL.backward(retain_graph=True)
            optimizer.step()
            if verbose:
                print(f'LL={-LL}, lr={lr}')
            else:
                if steps % 50 == 49:
                    print('.') # print dot and get to a new line
                else:
                    print('.', end="")  # print 50 dots in a row
        return -LL

########################################## MAIN ##############################################
if __name__ == "__main__":
    def one(x: torch.Tensor):
        return x.pow(0)

    def pow1(x: torch.Tensor):
        return x

    def pow2(x: torch.Tensor):
        return x.pow(2)

    # reg = GaussianRegression([torch.sin, torch.cos])
    # print(reg.simulate([0]))

    # Generating training data
    tr = GaussianRegression([one, pow1, pow2], means=[0,-1,0.5], st_devs=[0.3,0.2,0.1])
    x = torch.linspace(-2,3, steps=101)
    y_train = tr.simulate(x) # training set
    # y_train = torch.tensor(np.load('train_y.npy'))
    # print(f'Training values: {y_train}')
    plt.plot(x.tolist(), y_train.tolist(), '.', c='red', label='Data points')  # data points
    plt.axhline(c="grey", linewidth=0.5)
    plt.axvline(c="grey", linewidth=0.5)
    # Plotting the average
    xx=np.linspace(min(x), max(x), num=100)
    cf = tr.coefs_classic(x, y_train) # fitted coefficients in non-random regression model
    rt = tr.average(xx, cf)
    plt.plot(xx.tolist(), rt.tolist(), '-', c='green')
    plt.show()

    # # theoretical values:
    # tra = tr.average(x)
    # trd = tr.deviation(x)
    # print(f'Averages={tra} for x={x}')
    # print(f'deviations={trd}')

    # fitting:
    # True values of the parameters:
    # reg.means
    # tensor([0.0000, -1.0000, 0.5000])
    # reg.st_devs
    # tensor([0.3000, 0.2000, 0.1000])
    # define the model:
    reg = GaussianRegression([one, pow1, pow2], means=[0,0,0], st_devs=[1,1,1])
    reg.fit(x, y_train, algo=torch.optim.Adam)
    # If the starting point is the true point, the algo stays there
    # reg.fit(x, y_train, algo=torch.optim.Adam, start_means = [0, -1, 0.5], start_st_devs = [0.3, 0.2, 0.1])
    # reg.fit(x, y_train, params=[reg.means]) # if fitting means only
    print(f'means={reg.means.tolist()}')
    print(f'True means: [0, -1, 0.5]')
    print(f'st_devs={reg.st_devs.tolist()}')
    print(f'True st_devs: [0.3, 0.2, 0.1]')
    print(f'log-likelihood = {-reg.loglik(x, y_train).tolist()}')

    # plotting simulated data (grey) from the fitted model together with the original data (red)
    for i in range(5):
        plt.plot(x.tolist(), reg.simulate(x).tolist(), '.', c='grey')
    plt.plot(x.tolist(), y_train.tolist(), '.', c='red', label='Simulated points from fitted model') # data points
    plt.axhline(c="grey", linewidth=0.5)
    plt.axvline(c="grey", linewidth=0.5)
    # Plotting the average and the 90% confidence bounds
    def ci(r: GaussianRegression, x: torch.Tensor, alpha=0.1):
        ra = r.average(x)
        rd = r.deviation(x)
        q = -norm.ppf(alpha/2)
        cil = ra - q * rd
        cir = ra + q * rd
        return (ra, cil, cir)

    alpha = 0.1
    conf = ci(reg, xx, alpha)
    plt.plot(xx, conf[0].tolist(), '-', c='green', linewidth=2) # the expected line
    plt.plot(xx, conf[1].tolist(), '-', c='orange') # lower CI
    plt.plot(xx, conf[2].tolist(), '-', c='orange') # upper CI
    plt.show()

    # Assessing the model by the number of misses for the training set of CI's
    N = len(x)
    miss = np.zeros(N)
    cis = ci(reg, x, alpha)
    no_misses = 0
    for i in range(N):
        if (y_train[i] < cis[1][i]) or (y_train[i] > cis[2][i]):
            no_misses += 1
    # no_misses follows binomial Bin(len(x), alpha) distribution
    # alpha-CI for this distribution:
    lci = binom.ppf(alpha/2, N, alpha)
    uci = binom.ppf(1-alpha/2, N, alpha)
    if (no_misses >= lci) and (no_misses <= uci):
        print(f'{no_misses} is inside the confidence interval ({lci}, {uci}):')
        print(f'No contradiction with the fitted model at {100*(1-alpha)}% confidence level')
    else:
        print(f'{no_misses} is outside the confidence interval ({lci}, {uci}):')
        print(f'Number of misses is significantly at {100*(1-alpha)}% confidence level different from expected for the fitted model!')
