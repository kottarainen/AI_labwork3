import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Preparation of training data
# Use the MNIST dataset for classification
# Divide data into training and testing sets
# Normalize images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Perform image augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomAffine(0, scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data_augmented = datasets.MNIST(root='data', train=True, download=True, transform=train_transform)

# Visualize dataset images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axs[i].imshow(train_data[i][0].squeeze(), cmap='gray')
    axs[i].set_title(f'Label: {train_data[i][1]}')
plt.show()

# Visualize training images after augmentation
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axs[i].imshow(train_data_augmented[i][0].squeeze(), cmap='gray')
    axs[i].set_title(f'Label: {train_data_augmented[i][1]}')
plt.show()

# Create CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the model
model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# DataLoaders
train_loader = DataLoader(train_data_augmented, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Training the model
n_epochs = 10
valid_loss_min = np.Inf

for epoch in range(n_epochs):
    train_loss = 0.0

    # Train the model
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    
    # Calculate average training loss
    train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch+1}/{n_epochs} \tTraining Loss: {train_loss:.6f}')

# Save the model
torch.save(model.state_dict(), 'best_model.pth')

# Evaluation
model.eval()
test_loss = 0.0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = correct / len(test_loader.dataset)

print(f'Test Loss: {test_loss:.6f}')
print(f'Test Accuracy: {test_accuracy:.2%}')
