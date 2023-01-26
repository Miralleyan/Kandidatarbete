
import torch
  
# Creating a test tensor
x = torch.randint(1, 100, (100, 100))
  
# Checking the device name:
# Should return 'cpu' by default
print(x.device)
  
# Applying tensor operation
res_cpu = x ** 2
  
# Transferring tensor to GPU
x = x.to(torch.device('cuda'))
  
# Checking the device name:
# Should return 'cuda:0'
print(x.device)
  
# Applying same tensor operation
res_gpu = x ** 2
  
# Checking the equality
# of the two results
assert torch.equal(res_cpu, res_gpu.cpu())