# PyTorch DigitNet - CNN-Based Handwritten Digit Classification and Recognition
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# Define the CNN model in PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Initialize fc1 dynamically based on the conv output size
        self._to_linear = None
        self.convs(torch.randn(1, 1, 28, 28))
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def convs(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=1)


# Transformations for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Instantiate the model, define the loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}], Accuracy: {accuracy:.2f}%, Loss: {running_loss / len(train_loader):.4f}')
    return accuracy, running_loss / len(train_loader)


# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return all_labels, all_preds


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)

# Evaluate on test data
all_labels, all_preds = evaluate(model, test_loader)

# Confusion Matrix and Classification Report
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, zero_division=0))

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model with weights_only=True
loaded_model = SimpleCNN()
loaded_model.load_state_dict(torch.load('model.pth', weights_only=True))
loaded_model.to(device)
loaded_model.eval()

# Visualization of a test image and its predicted label
index = 6358
sample_image = test_dataset[index][0].numpy().squeeze()
sample_label = test_dataset[index][1]

with torch.no_grad():
    prediction = loaded_model(test_dataset[index][0].unsqueeze(0).to(device))
    predicted_label = prediction.argmax(dim=1).item()

plt.imshow(sample_image, cmap='gray')
plt.title(f"True Label: {sample_label}, Predicted Label: {predicted_label}")
plt.show()
