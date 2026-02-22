import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        super().__init__()
        # Deeper network with more capacity for quadratic approximation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        # Output layer: no activation for regression
        x = self.fcout(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, output_dim=10, input_height=32, input_width=32):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.ReLU()

        # Calculate flattened size after convolutions and pooling
        # After 3 pooling layers: height/8, width/8
        flat_dim = 128 * (input_height // 8) * (input_width // 8)

        # Fully connected layers
        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fcout = nn.Linear(128, output_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional block 1
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)

        # Convolutional block 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        # Convolutional block 3
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool3(x)

        # Flatten for dense layers
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.fc1(x)
        x = self.activation(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.activation(x)
        # x = self.dropout(x)

        x = self.fcout(x)
        return x


if __name__ == "__main__":
    # modelMLP = MultiLayerPerceptron(input_dim = 1, output_dim = 1, hidden_dim=16)
    # modelCNN = SimpleCNN(input_channels=3, output_dim=10, input_height=32, input_width=32)
    pass
