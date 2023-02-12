import torch


torch.manual_seed(1)

p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)

optimizer = torch.optim.Adam([p], lr=0.1)
epochs = 100

for epoch in range(epochs):
    grad = torch.autograd.grad(-p[0]*p[1]+p[2]*(p[0]+p[1]-1), p)
    grad = grad[0].requires_grad_()
    loss = torch.sum(torch.square(grad))
    grad2 = torch.autograd.grad(loss, grad)
    p.grad = grad2[0]
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}\tP = {p.tolist()}')
