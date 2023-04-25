from scipy import integrate
def f(*args):
    x, y, z = args
    return x*y*z

print(integrate.nquad(f,[[0,1], [0,1], [0,1]]))