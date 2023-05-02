import numpy as np
import torch
'''
blubb=np.load(f'../Finalized/test_data/data_1000_y_{0}.npy')
b=np.mean(blubb)
s=np.std(blubb)
print(b,s)
#print(blubb)

data=np.load(f'../Finalized/test_data/params.npy')
print(data[0][0])
print(data[1][1])
'''


'''


a=np.array([1,2,3])
b=np.array([0,1,7])
x=np.array([0.1,0.2,0.3])
print(a*x+b)
'''
x=np.linspace(-5,5,100)
print(x**2)
data = {
    'y': [0,1,3],
    'y_lin': 1,
    'y_sqr': 2,
    'y_nonNorm': 3
}
for k,v in data.items():
    print(np.array(k))
    print(v)
