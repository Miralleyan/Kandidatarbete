import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

x_values=[i for i in range(11)]
x_train=torch.Tensor(x_values)

y_values=[i for i in range(11)]
y_train=torch.Tensor(x_values)

values=torch.Tensor([x_values,y_values])

z_train=3*x_train+9*y_train+7



class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = 2        # takes variable 'x', "y" 
outputDim = 1       # takes variable 'z'
learningRate = 0.01 
epochs = 1000

model = linearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    inputs=values.t()
    labels=z_train.unsqueeze(-1)
     
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, labels)
    #print(loss)

    loss.backward()

    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


with torch.no_grad():
    predicted=model(inputs)
    print(predicted)



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(np.array(x_train.unsqueeze(-1)), np.array(y_train.unsqueeze(-1)),np.array(predicted),'--',label="Predictions",alpha=0.5)
ax.scatter(np.array(x_train.unsqueeze(-1)), np.array(y_train.unsqueeze(-1)),np.array(z_train.unsqueeze(-1)), c=np.array(z_train.unsqueeze(-1)))


'''
plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()

'''
