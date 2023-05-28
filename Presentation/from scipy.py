from scipy.stats import binom
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
# setting the values
# of n and p
n = 6
p = 0.5
# defining list of r values
r_values = list(range(n + 1))
# list of pmf values
dist = [binom.pmf(r, n, p) for r in r_values ]
# plotting the graph 
#plt.bar(r_values, dist)
#plt.show()
#print(sum(dist))

###
x=np.linspace(0,1)
plt.plot(x,beta.pdf(x,2,5))
plt.show()