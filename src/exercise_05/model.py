import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size=3072, output_dim=10, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(0.5))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the image (batch_size, channels, height, width) -> (batch_size, -1)
        x = x.view(x.size(0), -1)
        return self.model(x)


if __name__ == "__main__":
    model = MLPClassifier(input_size=3072, output_dim=10)
    print(model)
    # Test with a random batch
    x = torch.randn(4, 3, 32, 32)  # batch_size=4, channels=3, height=32, width=32
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {model(x).shape}")
