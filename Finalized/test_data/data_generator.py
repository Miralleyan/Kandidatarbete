import numpy as np
np.random.seed(0)
tests = 50

filename = 'data.npy'
means = np.array([])
stds = np.array([])

for i in range(tests):
    mean = np.random.normal(0,1,3)
    std = np.abs(np.random.normal(0,1,3))
    means = np.append(means, mean)
    stds = np.append(stds, std)
    for N in [100, 500, 1000]:
        a = np.random.normal(mean[0], std[0], N)
        b = np.random.normal(mean[1], std[1], N)
        c = np.random.normal(mean[2], std[2], N)

        x = np.linspace(-5, 5, N)
        y = a
        y_lin = a + x * b
        y_sqr = a + x * b + x**2 * c
        y_nonNorm = np.concatenate((a[0:N//2], b[0:N//2]))

        data = {
            'y': y,
            'y_lin': y,
            'y_sqr': y_sqr,
            'y_nonNorm': y_nonNorm
        }

        for k, v in data.items():
            file, ext = filename.split('.', 1)
            name = file + '_' + str(N) + '_' + k + '_' + str(i) + '.' + ext

            with open(name, 'wb') as f:
                np.save(f, np.array([x, v]).reshape(-1, 2))

params = np.array([means,stds])
# Saves means and stds in npy file in format: [[means],[stds]]
np.save('params.npy', params)
