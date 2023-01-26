import torch

p = torch.nn.Parameter(torch.randn(7))

opt = torch.optim.Adam(params=[p], lr=0.1)

for epoch in range(1000):
    opt.zero_grad()
    loss = p[0]*p[1] + p[2]*(p[0] + p[1] - 1) -p[3]*p[0] + p[4]*(p[0]-1) - p[5]*p[1] + p[6]*(p[1]-1)
    loss.backward()
    loss
    opt.step()
    with torch.no_grad():
        p[2:].clamp_(0)

print(p.detach())

    # pq + u1(p+q-1) + u2 (-p) + u3(p-1) + u4 (-q) + u5(q-1)
