import torch

# Modules
torch.nn.Linear(in_features, out_features)                          # x*A + b
torch.nn.LazyLinear(out_features)                                   # in_features is inferred
torch.nn.Bilinear(in_features1, in_features1, out_features)         # x1 * A * x2 + b

# Loss Functions
torch.nn.MSELoss()                                                  # Mean Squared Error
torch.nn.CrossEntropyLoss()                                         # God for classification

# Optimizer
opt = torch.optim.SGD()                                             # Stochastic Gradient Decent
opt.step()                                                          # Update parameters using gradients

# Dataloader
data_loader = torch.utils.data.DataLoader(data, batch_size)         # Reduce memory usage by bathing data

# Non-linear Activations
torch.nn.ReLU()                                                     # y = 0 if x < 0 else y = x
torch.nn.Sigmoid()                                                  # [0, 1]
torch.nn.Softmax()                                                  # scale elements to [0, 1] and sum = 1

# Tensor
x = torch.randn()                                                   # tensor filled with values from standard normal distribution
y = x.view(a,b,...)                                                 # reshapes x into size (a,b,...)
y = x.transpose(a,b)                                                # swaps dimensions a and b
y = x.unsqueeze(dim)                                                # tensor with added axis
y = x.squeeze()                                                     # removes all dimensions of size 1 (a,1,b,1) -> (a,b)
x.mm(y)                                                             # Matrix multiplication

# GPU
# Note: I had to install torch from commandline to access gpu
x = x.to(device)                                                    # Copy tensor to a device (gpu, cpu)


