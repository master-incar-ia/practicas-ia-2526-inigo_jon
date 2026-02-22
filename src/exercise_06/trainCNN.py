from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import ECGDataset
from .model import SimpleCNN


def get_device(force: str = "auto") -> torch.device:
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(output_folder: Path, device: torch.device):

    # Dataset
    dataset_train = ECGDataset(train=True)
    dataset_test = ECGDataset(train=False)

    # Convert to torch tensors
    dataset_train.X = torch.tensor(dataset_train.X, dtype=torch.float32)
    dataset_train.Y = torch.tensor(dataset_train.Y, dtype=torch.float32)

    dataset_test.X = torch.tensor(dataset_test.X, dtype=torch.float32)
    dataset_test.Y = torch.tensor(dataset_test.Y, dtype=torch.float32)

    # Split train/validation
    train_size = int(0.8 * len(dataset_train))
    val_size = len(dataset_train) - train_size

    train_dataset, val_dataset = random_split(
        dataset_train,
        [train_size, val_size]
    )

    # Normalización usando SOLO entrenamiento
    train_indices = train_dataset.indices
    x_train = dataset_train.X[train_indices]

    x_mean = x_train.mean()
    x_std = x_train.std()
    if x_std == 0:
        x_std = 1.0

    dataset_train.X = (dataset_train.X - x_mean) / x_std
    dataset_test.X = (dataset_test.X - x_mean) / x_std

    np.savez(
        str(output_folder / "norm_params.npz"),
        x_mean=float(x_mean),
        x_std=float(x_std),
    )

    # DataLoaders
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=pin_memory,
    )

    # Dimensiones reales del ECG
    input_dim = dataset_train.X.shape[1]
    output_dim = 2

    # Modelo
    model = SimpleCNN(
        input_channels=1,
        output_dim=output_dim,
        input_height=dataset_train.Y.shape[1],
        input_width=dataset_train.X.shape[1]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    best_val_loss = float("inf")
    best_model_path = output_folder / "best_model.pth"

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):

        # ---- TRAIN ----
        model.train()
        train_loss = 0

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in val_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"Train Loss={train_loss:.4f} "
                f"Val Loss={val_loss:.4f}"
            )

    print("Best validation loss:", best_val_loss)

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(output_folder / "loss_plot.png")


if __name__ == "__main__":

    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device("auto")
    print("Using device:", device)

    train_model(output_folder, device)