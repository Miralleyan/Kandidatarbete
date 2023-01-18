import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

batchSize = 20000
learningRate = 0.01
epochs = 1
cudaEnabled = True

dev = 'cuda:0' if cudaEnabled else 'cpu'
device = torch.device(dev)

# Transform PIL image into a tensor. The values are in the range [0, 1]
t = transforms.ToTensor()

# Load datasets for training and apply the given transformation.
mnist_training = datasets.MNIST(root='data', train=True, download=True, transform=t)
mnist_val = datasets.MNIST(root='data', train=False, download=True, transform=t)

model = torch.nn.Sequential(
    torch.nn.Flatten(1, -1),         # Convert to ({batchSize}, 784) tensor
    torch.nn.Linear(28*28, 256),    # Hidden layer, 256 neurons
    torch.nn.ReLU(),                # Rectify negative values
    torch.nn.Linear(256, 10)        # Output layer
)
model = model.to(device)
lossFn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learningRate)

# Specify a data loader which returns 500 examples in each iteration.
train_loader = torch.utils.data.DataLoader(mnist_training, batch_size=batchSize, shuffle=True)

losses = []
scaler = torch.cuda.amp.GradScaler(enabled=cudaEnabled)

# Train the model
for epoch in range(epochs):
    for imgs, labels in train_loader:

        if cudaEnabled:
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                imgs, labels = imgs.to(device), labels.to(device)
                predictions = model(imgs)
                assert predictions.dtype is torch.float16
                loss = lossFn(predictions, labels)
                assert loss.dtype is torch.float32

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(imgs)
            loss = lossFn(predictions, labels)

            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        losses.append(float(loss))

    print(f"Epoch: {epoch}, Loss: {float(loss):.5f}")
    
# Plot learning curve
plt.plot(losses)
plt.draw()

exit()

# Load all 10000 images from validation set
n = 10000
val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=n)
images, labels = iter(val_loader).next()

# Apply model to predict digit in image
with torch.no_grad():
    predictions = model(images)

# Determine highest score for each row
predicted_classes = torch.argmax(predictions, dim=1)

# Calculate accuracy
accuracy = sum(predicted_classes.numpy() == labels.numpy()) / n
print(f"Accuracy of model: {accuracy*100:.2f}%")

# Plot some digits.
    # cols = 8
    # rows = 2

    # fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    # for i, ax in enumerate(axes.flatten()):
    #     image, label = mnist_training[i]          # returns PIL image with its labels
    #     ax.set_title(f"Label: {label}")
    #     ax.imshow(image.squeeze(0), cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension

plt.show()