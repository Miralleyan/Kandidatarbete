import torch
import pytorch_measure as pm

def main():
    a=torch.tensor([0.1,0.1,0.3,0.1,0.4])
    b=torch.tensor([1.,2.,3.,4.,5.])

    c=pm.Pytorch_measure(b,a)

    print(c.put_mass(0.2, 1))
    print(c)

if __name__ == '__main__':
    main()