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


# CNN para señales 1D (ECG)
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, output_dim=2, input_length=1000):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Calcular tamaño tras convoluciones y pooling
        flat_dim = input_length
        for _ in range(3):
            flat_dim = flat_dim // 2
        flat_dim = 128 * flat_dim

        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fcout = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: (batch, length) o (batch, 1, length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fcout(x)
        return x


if __name__ == "__main__":
    # modelMLP = MultiLayerPerceptron(input_dim = 1, output_dim = 1, hidden_dim=16)
    # modelCNN = SimpleCNN(input_channels=3, output_dim=10, input_height=32, input_width=32)
    pass
