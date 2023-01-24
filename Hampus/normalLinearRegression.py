import numpy as np
import matplotlib.pyplot as plt

# Dummy data
x = np.array(range(50))
y = 2 + 3 * x + np.random.normal(0, 5, len(x))

# Learning parameters
a = 0
b = 0

L = 0.0001  # Learning rate
epochs = 1000  # Training epochs

n = len(x)  # Elements in x

# Gradient decent
for _ in range(epochs):
    y_pred = b + a * x  # Predict y with current a & b
    D_a = (-2/n) * sum(x * (y - y_pred))  # Derivative wrt a
    D_b = (-2/n) * sum(y - y_pred)  # Derivative wrt b
    a = a - L * D_a  # Update a
    b = b - L * D_b  # Update b
print(f'{b:.3f} + {a:.3f}.*x')

# Plot results
plt.clf()
plt.scatter(x, y, s=20, marker='.', label='True Data', alpha=0.7)
plt.plot(x, y_pred, 'r--',  label='Prediction', alpha=0.8)
plt.legend(loc='best')
plt.show()
