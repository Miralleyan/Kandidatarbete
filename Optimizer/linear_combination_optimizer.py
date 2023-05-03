import torch
import numpy as np
import matplotlib.pyplot as plt
import time

class Optimizer():
    def __init__(self, x, y, order=3, n=3, ax = False):
        self.order = order
        self.n = n
        self.x = x
        self.y = y
        if ax == False:
            self.h = [self.create_lambda(i) for i in range(self.order)]
        else:
            self.h = [self.create_lambda(1)]
        h_all_data = [h(x) for h in self.h]
        self.h_all = torch.transpose(torch.stack(h_all_data, 0), 0, 1)
        self.mu = torch.tensor([0. for _ in range(order)], dtype=float, requires_grad=True)
        self.sigma = torch.tensor([1. for _ in range(order)], dtype=float, requires_grad=True)
        self.beta = [self.mu, self.sigma]
    
    def create_lambda(self, i):
        if i == 0:
            return lambda u: u*0+1
        return lambda u:u**i
    
    def optimize(self, epochs=300, lr=0.1, print_frequency = 10, test = False):
        optimizer = torch.optim.Adam(self.beta,lr=lr, maximize=True)
        old_loss = float('inf')
        cur_epoch = epochs
        t1 = time.time()
        t2 = float('inf')     # In case optimization doesnt terminate before max epochs are reached
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.log_lik(self.y, self.beta, self.h_all)
            loss.backward()
            optimizer.step()
            if epoch%print_frequency==0:
                print(epoch, "mu:", self.mu.detach().numpy(), "sigma:", self.sigma.detach().numpy(), "loss:", loss)
            if torch.abs(loss-old_loss) < 1e-15:
                t2 = time.time()
                cur_epoch = epoch
                break
            old_loss = loss
        self.mu_optim = self.beta[0].detach().numpy()
        self.sigma_optim = self.beta[1].detach().numpy()
        print(epoch, "mu:", self.mu_optim, "sigma:", self.sigma_optim)
        if test == False:
            return [self.m(self.mu, self.h_all[i,:]).detach().numpy().tolist() for i in range(self.x.size(dim=0))], [(self.sigma_2(self.sigma, self.h_all[i,:])**0.5).detach().numpy().tolist() for i in range(self.x.size(dim=0))]
        else:
            if t2 != float('inf'):
                return [self.m(self.mu, self.h_all[i,:]).detach().numpy().tolist() for i in range(self.x.size(dim=0))], [(self.sigma_2(self.sigma, self.h_all[i,:])**0.5).detach().numpy().tolist() for i in range(self.x.size(dim=0))], cur_epoch, t2-t1
            else:
                return [self.m(self.mu, self.h_all[i,:]).detach().numpy().tolist() for i in range(self.x.size(dim=0))], [(self.sigma_2(self.sigma, self.h_all[i,:])**0.5).detach().numpy().tolist() for i in range(self.x.size(dim=0))], cur_epoch, None


    def m(self, mu, h_x):
        return (mu * h_x).sum()

    def sigma_2(self, sigma, h_x):
        return (sigma * h_x).pow(2).sum()

    def sigma_2_regular_exp(self, sigma, h_x):
        return ((sigma * h_x)**2).sum()

    def log_normal_pdf(self, y, mu, sigma_2):
        return -1/2*(torch.log(2*np.pi*(sigma_2)) + ((y-mu)**2/sigma_2))

    def log_lik(self, y, beta, h_all):
        return sum([self.log_normal_pdf(y_i, self.m(beta[0], h_all[i]), self.sigma_2(beta[1], h_all[i])) for i, y_i in enumerate(y)])

    def get_plot_CI(self, mu, sigma, n=3):
        y_new = []
        sigma_upper = []
        sigma_lower = []
        sample_x = []
        sample_y = []

        for i in range(len(self.x)):
            h_x = [h_i.item() for h_i in self.h_all[i]]
            y_new.append(self.m(mu, h_x))
            s = self.sigma_2_regular_exp(sigma, h_x)**0.5
            sigma_upper.append(y_new[-1] + 2*s)
            sigma_lower.append(y_new[-1] - 2*s)
            sample = torch.normal(float(y_new[-1]), float(s), size = (1, n))
            for j in range(n):
                sample_x.append(self.x[i].item())
                sample_y.append(sample[0][j].item())
        return y_new, sigma_upper, sigma_lower
    
    def visualize(self):
        y_new, sigma_upper, sigma_lower = self.get_plot_CI(self.mu_optim, self.sigma_optim)
        markersize = 20
        plt.scatter(self.x, self.y, sizes=[markersize]*len(self.x))
        plt.plot(self.x, y_new, 'r-')
        plt.plot(self.x, sigma_upper, 'r--')
        plt.plot(self.x, sigma_lower, 'r--')
        plt.show()