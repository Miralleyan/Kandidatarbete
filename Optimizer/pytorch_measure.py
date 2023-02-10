import torch

class pytorch_measure():
    def __init__(self,locations,weights):
        """
        Group 1: 
        Group 2: 
        """
        self.locations = torch.nn.Parameter(locations)
        self.weights = torch.nn.Parameter(weights)
        pass

    def total_mass(self):
        """
        Responsibility:
        """
        pass

    def total_variation(self):
        """
        Responsibility:
        """
        pass

    def support(self):
        """
        Responsibility:
        """
        pass

    def positive_part(self):
        """
        Responsibility:
        """
        pass

    def negative_part(self):
        """
        Responsibility:
        """
        pass

    def put_mass(self):
        """
        Responsibility:
        """
        pass

    def take_mass(self):
        """
        Responsibility:
        """
        pass

    def sample(self):
        """
        Responsibility:
        """
        pass