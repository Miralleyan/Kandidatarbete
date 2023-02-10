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

    def put_mass(self,epsilon) -> bool:
        """
        :param:
        :returns: True if all mass could be placed, False otherwise
        Responsibility: Johan (Inte färdig)
        """
        dic={self.weights[i].item() :i for i in range(len(self.weights))}
        i=dic[min(dic)]
        if (self.weights[i]).item()+epsilon > 1:
            epsilon-=(1-self.weights[i].item())
            self.weights[i]+=(1-self.weights[i].item())
            return epsilon
        else:
            self.weights[i]+epsilon
            return 0,dic
        

    def take_mass(self, mass, location_index) -> float:
        """
        Responsibility: Samuel
        In current form, this method takes mass from a specified location, s.t. the location still
        has non-negative mass and returns how much mass is left to take.
        :param: mass left to take, index of location to take at
        :returns: mass left to remove from measure after removing from specified location
        """
        mass_left = mass
        self.weights[location_index] = max(self.weights[location_index] - mass, 0)
        if mass > self.weights[location_index]:
            mass_left -= self.weights[location_index]
            self.weights[location_index] == 0
        else:
            mass_left = 0
            self.weights[location_index] -= mass
        return mass_left
            
            

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


a=torch.tensor([0.1,0.,0.3,0.,0.4])
b=torch.tensor([1.,2.,3.,4.,5.])

c=pytorch_measure(b,a)

print(c.put_mass(0.2))