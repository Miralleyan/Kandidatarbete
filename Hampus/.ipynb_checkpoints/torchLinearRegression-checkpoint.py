import torch
import numpy as np
import matplotlib.pyplot as plt

# Dummy data
x = np.linspace(0, 50)
y = 2 + 3 * x + np.random.normal(0, 5, len(x))

# Convert to tensor, transpose to 50x1 and convert to float
x = torch.from_numpy(x).unsqueeze(-1).float()
y = torch.from_numpy(y).unsqueeze(-1).float()

L = 0.001  # Learning rate
epochs = 10000  # Training epochs

# Create model
# Linear layer with 1 input, 1 output, aka 1 neuron
model = torch.nn.Linear(1, 1)

# Create loss function or "criterion"
# Using Mean Square Error
criterion = torch.nn.MSELoss()

# Create optimizer
# Using Stochastic Gradient Descent
# Learning rate and model parameters as input
optimizer = torch.optim.SGD(lr=L, params=model.parameters())

# Train network
for _ in range(epochs):
    optimizer.zero_grad()  # Reset gradients every epoch
    y_pred = model(x)  # Evaluate model with current parameters
    loss = criterion(y_pred, y)  # Calculate loss
    loss.backward()  # Calculate gradient
    optimizer.step()  # Update parameters

# Read model parameters
a = model.weight.item()
b = model.bias.item()

print(f'{b:.3f} + {a:.3f}.*x')

# Plot results
plt.clf()
plt.scatter(x.tolist(), y.tolist(), s=20, marker='.', label='True Data', alpha=0.7)
plt.plot(x.tolist(), y_pred.tolist(), 'r--',  label='Prediction', alpha=0.8)
plt.legend(loc='best')
plt.show()


# Custom class example
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.b + self.a * x

    def string(self):
        return f'y = {self.a.item():.4f} + {self.b.item():.4f} * x'
