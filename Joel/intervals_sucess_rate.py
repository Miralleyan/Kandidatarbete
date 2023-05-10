from scipy.stats import beta
import numpy as np

disc_tau = np.array([0, 0.52, 0.48, 0.84, 0, 0.58, 0.66, 0.86, 0, 0.7, 0.9, 0.92])
poly_tau = np.array([1, 0.98, 1, 0.98, 0.98, 0.88, 0.92, 0.84, 1, 0.76, 0.74, 0.7])
linj_tau = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.98, 1])

for tau in [disc_tau, poly_tau, linj_tau]:
    for t in tau:
        print(beta.interval(0.95, 1 + 50 * t, 1 + 50 * (1 - t)))
    print('---')

