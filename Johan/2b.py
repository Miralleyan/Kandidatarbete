import torch
#Får olika svar med den här beroende på startvärdet

p0 = torch.tensor(0.6, requires_grad=True)
p = p0
opt = torch.optim.Adam([p], lr=0.1)
steps = 100

for step in range(steps):
    opt.zero_grad()
    var=p*(1-p)
    var.backward()
    opt.step()
    with torch.no_grad():
        p.clamp_(0,1)
    if step % 10 == 9:
        print(f'Step {step+1: 2}: p={p.item(): 0.4f} and variance is {var.item(): 1.4f}')




#Får bara ena lösningen med den här då jag inte själv väljer startvärde
def var(p):
    return p*(1-p)
from scipy.optimize import minimize_scalar
'''
res = minimize_scalar(var,bounds=(0,1), method="bounded")
print(res.x,res.fun)
print(res)

'''
#Gives different answers depending on startvalues
from torchmin import minimize_constr
'''
x0=torch.tensor(0.6)
res=minimize_constr(var, x0,bounds={"lb":0,"ub":1},tol=1e-9)
print(res.x,res.fun)
'''
