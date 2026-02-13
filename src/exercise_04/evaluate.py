from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import CIFAR10Dataset
from .model import SimpleCNN

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def evaluate_and_plot(loader, model, dataset_name, output_folder, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average="weighted")
    recall = recall_score(all_targets, all_predictions, average="weighted")
    f1 = f1_score(all_targets, all_predictions, average="weighted")

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

    print(f"Evaluation metrics for {dataset_name} dataset:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
    )
    plt.title(f"Confusion Matrix for {dataset_name} dataset")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{dataset_name}_confusion_matrix.png")
    plt.show(block=False)
    plt.close()

    return metrics


def save_metrics_as_picture(metrics, filepath):
    # Create a DataFrame
    df = pd.DataFrame(metrics)

    # Round the values to 3 decimal places
    df = df.round(3)

    # Plot the table and save as an image
    fig, ax = plt.subplots(figsize=(8, 2))  # set size frame
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc="center", loc="center"
    )

    # Save the plot as an image
    plt.savefig(filepath)


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data normalization for CIFAR-10
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR-10 datasets
    dataset_train = CIFAR10Dataset("./data", train=True, transform=transform)
    dataset_test = CIFAR10Dataset("./data", train=False, transform=transform)

    # Split training set into train and validation
    train_size = int(0.85 * len(dataset_train))
    val_size = len(dataset_train) - train_size
    train_dataset, val_dataset = random_split(dataset_train, [train_size, val_size])

    # Create DataLoaders for the datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    # Load the best model weights
    model = SimpleCNN(input_channels=3, output_dim=10, input_height=32, input_width=32).to(device)
    best_model_path = output_folder / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Warning: Model weights not found at {best_model_path}")

    metrics = {}
    # Evaluate and plot for train, validation and test datasets
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder, device)
    metrics["validation"] = evaluate_and_plot(
        val_loader, model, "validation", output_folder, device
    )
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder, device)

    # save metrics as csv
    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")

    # Save the metrics as an image
    save_metrics_as_picture(metrics, output_folder / "metrics.png")

    print("Evaluation complete!")
