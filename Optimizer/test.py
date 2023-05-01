import numpy as np

blubb=np.load('../Finalized/test_data/data_100_y_lin.npy')
print(blubb.reshape(1,-2)[0][:100])
print(blubb.reshape(1,-2)[0][100:])
      
'''
data = {
    'y': 0,
    'y_lin': 1,
    'y_sqr': 2,
    'y_nonNorm': 3
}
for k,v in data.items():
    print(k)
    print(v)
    '''