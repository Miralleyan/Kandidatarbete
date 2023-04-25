import numpy as np

file_name = "test_data.npy"

with open(file_name, 'wb') as f:
    np.save(f, np.array([1, 2]))
    np.save(f, np.array([1, 3]))
with open(file_name, 'rb') as f:
    a = np.load(f)
    b = np.load(f)
print(a, b)