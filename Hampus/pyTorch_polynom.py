import matplotlib.pyplot as plt
import NeuralNetworkModules as nnm
import torch
import math

# Training parameters
learning_rate = 1e-3
epochs = 1600
verbose = False # Print progress
device = torch.device('cpu')

# Create learning set
x = torch.linspace(-math.pi, math.pi, 2000).to(device)
y = torch.sin(x).to(device)

model = nnm.Polynom(3).to(device)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


for t in range(epochs):

    # Forward pass: evaluate model
    y_pred = model(x, device)
   
    # Compute loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99 and verbose:
        print(t, loss.item())
    
    # Backward pass: Compute new gradients
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()

# Get layer weights and biases to print
linear_layer = model.linear
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

# Convert from tensor to array
    # x = x.data.numpy()
    # y = y.data.numpy()
    # y_pred = y_pred.data.numpy()

# Plot data and predicted
    # plt.clf()
    # plt.plot(x, y, '-', label='True data', alpha=0.8)
    # plt.plot(x, y_pred, '--', label='Predictions', alpha=0.8)
    # plt.legend(loc='best')
    # plt.show()