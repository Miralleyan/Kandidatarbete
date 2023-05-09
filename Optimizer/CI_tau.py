import scipy
import math

def CI(tau):
    bound=[]
    for t in tau:
        a=1.96*math.sqrt(t*(1-t))/math.sqrt(50)
        bound.append([t-a,t+a])
    return bound


OFDM=[0,0.52,0.48,0.84,0,0.58,0.66,0.86,0,0.7,0.9,0.92]
Poly=[1,0.98,1,0.98,0.98,0.88,0.92,0.84,1,0.75,0.74,0.7]
Linear=[1,1,1,1,1,1,1,1,1,1,0.98,1]
print(CI(OFDM))
print(CI(Poly))
print(CI(Linear))