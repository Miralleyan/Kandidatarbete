import matplotlib.pyplot as plt
import NeuralNetworkModules as nnm
import torch

# create dummy data for training
x_values = list(range(11))
x_train = torch.Tensor(x_values).unsqueeze(-1)

y_values = [2*i + 1 for i in x_values]
y_train = torch.Tensor(y_values).unsqueeze(-1)

learningRate = 0.01 
epochs = 500

model = nnm.LinearRegression()

loss_Fn = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = loss_Fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % (epochs/10) == 0:
        print(f'epoch: {epoch}, loss: {loss.item():.5f}')

with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(x_train)

print('\n' + model.string())

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()