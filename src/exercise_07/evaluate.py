from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, random_split

from ..exercise_07.dataset import ECGDataset
from ..exercise_07.model import LSTMClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def evaluate_and_plot(loader, model, dataset_name, output_folder, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Convertir a float32
            inputs = inputs.float()
            targets = targets.float()
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(torch.argmax(targets, dim=1).cpu().numpy())

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["shockable", "not_shockable"],
        yticklabels=["shockable", "not_shockable"],
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
    torch.manual_seed(42)

    # Cargar datasets
    dataset_train = ECGDataset(train=True)
    dataset_test = ECGDataset(train=False)

    # Normalización
    norm_path = output_folder / "norm_params.npz"
    if norm_path.exists():
        try:
            params = np.load(str(norm_path))
            x_mean = float(params["x_mean"])
            x_std = float(params["x_std"])
            dataset_train.X = (dataset_train.X - x_mean) / x_std
            dataset_test.X = (dataset_test.X - x_mean) / x_std
        except Exception:
            pass

    # DataLoaders
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    # Cargar modelo
    input_length = dataset_train.X.shape[1]
    model = LSTMClassifier(input_size=1, hidden_size=16, num_layers=2, output_dim=2, dropout=0.5)
    model.load_state_dict(torch.load(output_folder / "best_model_lstm.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics = {}
    metrics["train"] = evaluate_and_plot(train_loader, model, "train", output_folder, device)
    metrics["test"] = evaluate_and_plot(test_loader, model, "test", output_folder, device)

    pd.DataFrame(metrics).to_csv(output_folder / "metrics.csv")
    save_metrics_as_picture(metrics, output_folder / "metrics.png")
    print("Evaluation complete!")
