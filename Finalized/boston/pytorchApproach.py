"""
General PyTorch approach to linear regression fitting
- implemented on data about Boston housing prices
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


def dataframe_to_arrays(data):
    # Make a copy of the original dataset
    dataframe = data.copy(deep=True)

    # Extract input & outputs as numpy arrays
    input_array = dataframe[input_cols].to_numpy()
    target_array = dataframe[output_col].to_numpy()
    return input_array, target_array.reshape((len(data.index), 1))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


dataset = pd.read_csv('boston_housing.csv')
dataset.drop(['chas'], axis=1, inplace=True)
num_rows = len(dataset.index)
input_cols = list(dataset.columns)
print(f'The total number of rows in the dataframe is: {num_rows}')
print(f'Dataframe columns: {input_cols} \n')

# Visualization and information of dataset
# sns.pairplot(dataset,x_vars=['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'],
# y_vars=['medv'], height=7, aspect=0.5, kind='scatter')
# sns.pairplot(dataset)
# sns.heatmap(dataset.corr(), annot=True)
# print(dataset.corr().medv.sort_values(ascending=False))
# sns.displot(dataset.medv)
# plt.savefig('pair_plot.png', dpi=300)
# plt.show()

input_cols = dataset.columns[0:12]
output_col = [dataset.columns[-1]]
inputs_size = len(input_cols)
outputs_size = len(output_col)

# Create PyTorch Dataset
inputs_array, targets_array = dataframe_to_arrays(dataset)
inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)
dataset = TensorDataset(inputs, targets)

# Split dataset into validation and training subsets
val_percent = 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Create PyTorch DataLoaders
batch_size = 16
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


# Create Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, xb):
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calcuate loss
        loss = F.l1_loss(input=out, target=targets, size_average=None, reduce=None, reduction='mean')
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss = F.l1_loss(input=out, target=targets, size_average=None, reduce=None, reduction='mean')
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result['val_loss']))


model = LinearRegressionModel(inputs_size, outputs_size)

# Loss on validation set before training
result = evaluate(model, val_loader)
print(f'Result before training: {result}')
print(f'Parameters before training: {list(model.parameters())}\n')

# Train the model
epochs = 2000
lr = 1e-7
history1 = fit(epochs, lr, model, train_loader, val_loader)
print('\n')
print(f'Parameters after training: {list(model.parameters())}\n')


# Making predictions
def make_prediction(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    return f'Input: {input}\n' \
           f'Target: {target}\n' \
           f'Prediction: {prediction}'


input, target = val_ds[11]
print(make_prediction(input, target, model))
