import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import NumModel   # import class from separate file

# ---------- Load & Preprocess the Data ----------
transform = transforms.Compose([
    transforms.ToTensor(),              # converts to tensor (0–1)
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Display the first image
image, label = train_dataset[0]
plt.imshow(image.view(28, 28), cmap="gray")
plt.title(f"Label: {label}")
plt.show()

# ---------- Build the Neural Network ----------
model = NumModel()
print(model)

# ---------- Loss & Optimizer ----------
criterion = nn.CrossEntropyLoss()   # no one-hot needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------- Train the Neural Network ----------
epochs = 3
train_accuracy = []
val_accuracy = []

for epoch in range(epochs):
    correct = 0
    total = 0

    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    train_accuracy.append(acc)
    print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {acc:.4f}")

# ---------- Evaluate the Model ----------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ---------- Visualize Training ----------
plt.plot(train_accuracy)
plt.title("Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

# ---------- Save the Model ----------
torch.save(model.state_dict(), "num_model.pth")

# ---------- Make Predictions ----------
test_images, test_labels = next(iter(test_loader))
outputs = model(test_images)
_, preds = torch.max(outputs, 1)

for i in range(5):
    plt.imshow(test_images[i].view(28, 28), cmap="gray")
    plt.title(f"Predicted: {preds[i].item()}, Actual: {test_labels[i].item()}")
    plt.show()
