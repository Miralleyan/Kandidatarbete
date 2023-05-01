import numpy as np

blubb=np.load(f'../Finalized/test/data_1000_y_lin_{1}.npy')
print(blubb[0])
#print(blubb)

data=np.load(f'../Finalized/test/params.npy')
print(data[0][3])
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