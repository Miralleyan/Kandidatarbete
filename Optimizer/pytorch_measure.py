import torch

class Pytorch_measure:
    def __init__(self,locations,weights):
        """
        Group 1: 
        Group 2: 
        """
        self.locations = torch.nn.parameter.Parameter(locations)#Input must be tensors
        self.weights = torch.nn.parameter.Parameter(weights)
    
    def __str__(self) -> str:
        """
        Responsibilty: Filip
        """
        return self.locations.__str__() + self.weights.__str__()

    def __repr__(self) -> str:
        """
        Responsibilty: Filip
        """
        return self.__str__()

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

    # Return support of measure
    def support(self):
        """
        Responsibility: Johan
        """
        return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item()!=0])

    # Return support of measure with positive mass
    def positive_part(self):
        """
        Responsibility: Samuel
        """
        return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item() > 0])

    # Return support of measure with negative mass
    def negative_part(self):
        """
        Responsibility: Johan
        """
        return torch.tensor([self.locations[i].item() for i in range(len(self.locations)) if self.weights[i].item()<0])

    def put_mass(self, mass, location_index) -> float:
        """
        In current form, this method puts mass at a specified location, s.t. the location still
        has mass less at most 1 and returns how much mass is left to distribute.
        :param: mass to put, index of location to put at
        :returns: mass left to add to measure after adding at specified location
        Responsibility: Johan, Samuel (Inte färdig)
        """
        # Johan's implementation
        # dic={self.weights[i].item() :i for i in range(len(self.weights))}
        # i=dic[min(dic)]
        # if (self.weights[i]).item()+epsilon > 1:
        #     epsilon-=(1-self.weights[i].item())
        #     self.weights[i]+=(1-self.weights[i].item())
        #     return epsilon
        # else:
        #     self.weights[i]+epsilon
        #     return 0,dic
        with torch.no_grad():
            if self.weights[location_index].item() + mass > 1:
                mass_distributed = 1 - self.weights[location_index].item()
                self.weights[location_index] = 1
            else:
                self.weights[location_index] += mass
                mass_distributed = mass
        return mass_distributed

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

    def sample(self, size):
        """
        Responsibility: Samuel
        """
        sampling = torch.rand(size).sort(key = lambda x: x.item())
        cur_sample = []
        i = 0
        for j in range(len(self.weights)):
            count = 0
            while i < len(self.weights) and sampling[i] < self.weights[j].item():
                i += 1
                count += 1
            cur_sample.append([location[j].item()]*count)
        return torch.tensor(cur_sample)
            

    def step(self):
        """
        Responsibility: Hampus
        Takes one optimization step with algoritm
        """
        pass


def main():
    a=torch.tensor([0.1,0.1,0.3,0.1,0.4])
    b=torch.tensor([1.,2.,3.,4.,5.])

    c=Pytorch_measure(b,a)

    print(c.put_mass(0.2, 1))
    print(c)

if __name__ == "__main__":
    main()