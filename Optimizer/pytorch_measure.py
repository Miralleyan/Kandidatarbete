import torch

class pytorch_measure:
    def __init__(self,locations,weights):
        """
        Group 1: 
        Group 2: 
        """
        self.locations = torch.nn.parameter.Parameter(locations)#Input must be tensors
        self.weights = torch.nn.parameter.Parameter(weights)

    def total_mass(self):
        """
        Responsibility: Johan
        """
        return sum(self.weights).item()

    def total_variation(self):
        """
        Responsibility: Samuel
        """
        return sum(abs(self.weights)).item()

    def support(self):
        """
        Responsibility: Johan
        """
        return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item()!=0])


    def positive_part(self):
        """
        Responsibility: Samuel
        """
        return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item() > 0]), \
            torch.tensor([weight.item() for weight in self.weights if weight.item() > 0])

    def negative_part(self):
        """
        Responsibility: Johan
        """
        return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item()<0])

    def put_mass(self) -> bool:
        """
        
        :param:
        :returns: True if all mass could be placed, False otherwise
        Responsibility: Johan
        """
        pass

    def take_mass(self) -> bool:
        """
        Responsibility:
        :param:
        :returns: True if all mass could be taken, False otherwise
        """
        pass

    def sample(self):
        """
        Responsibility:
        """
        pass

    def step(self):
        """
        Responsibility: Hampus
        Takes one optimization step with algoritm
        """
        pass

a=torch.tensor([-1.,0.,-2.,0.,10.])
b=torch.tensor([1.,2.,3.,4.,5.])

c=pytorch_measure(b,a)

print(c.negative_part())