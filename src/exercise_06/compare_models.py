from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from dataset import ECGDataset
from model import SimpleCNN, MultiLayerPerceptron

def evaluate_and_plot(loader, model, model_name, dataset_name, output_folder, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = inputs.float()
            targets = targets.float()
            if isinstance(model, SimpleCNN) and inputs.ndim == 2:
                inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(torch.argmax(targets, dim=1).cpu().numpy())
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
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
    print(f"[{model_name}] Evaluation metrics for {dataset_name} dataset:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
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
    plt.title(f"Confusion Matrix for {model_name} ({dataset_name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{model_name}_{dataset_name}_confusion_matrix.png")
    plt.show(block=False)
    plt.close()
    return metrics

def main():
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(42)
    dataset_train = ECGDataset(train=True)
    dataset_test = ECGDataset(train=False)
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
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CNN
    input_length = dataset_train.X.shape[1]
    cnn_model = SimpleCNN(input_channels=1, output_dim=2, input_length=input_length)
    cnn_model.load_state_dict(torch.load(output_folder / "best_model_cnn.pth"))
    cnn_model.to(device)
    # MLP
    input_dim = dataset_train.X.shape[1]
    mlp_model = MultiLayerPerceptron(input_dim=input_dim, output_dim=2, hidden_dim=64)
    mlp_model.load_state_dict(torch.load(output_folder / "best_model_mlp.pth"))
    mlp_model.to(device)
    metrics = {}
    metrics["CNN_train"] = evaluate_and_plot(train_loader, cnn_model, "CNN", "train", output_folder, device)
    metrics["CNN_test"] = evaluate_and_plot(test_loader, cnn_model, "CNN", "test", output_folder, device)
    metrics["MLP_train"] = evaluate_and_plot(train_loader, mlp_model, "MLP", "train", output_folder, device)
    metrics["MLP_test"] = evaluate_and_plot(test_loader, mlp_model, "MLP", "test", output_folder, device)
    pd.DataFrame(metrics).to_csv(output_folder / "metrics_compare.csv")
    print("Comparison complete!")

if __name__ == "__main__":
    main()
