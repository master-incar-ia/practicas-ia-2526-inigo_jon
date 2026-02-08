from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset


class NoisyRegressionDataset(Dataset):
    def __init__(self, noise_std=20, size=100, seed=42):
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(0, 100, size=(size,))
        self.delta = rng.normal(0, noise_std, size=(size,))
        self.y = -3 * self.x * self.x + 5 * self.x + self.delta

        # Create a DataFrame for visualization
        df = pd.DataFrame(data=np.array([self.x, self.y]).transpose(), columns=["x", "y"])
        self.df = df

        # Reshape for PyTorch compatibility and store as float32
        self.x = self.x.reshape((-1, 1)).astype(np.float32)
        self.y = self.y.reshape((-1, 1)).astype(np.float32)

        # Preserve a copy of the raw (unnormalized) inputs for later denormalization
        self.x_raw = self.x.copy()

    def plot(self, filepath):
        ax = sns.scatterplot(self.df, x="x", y="y")
        ax.set_title("Synthetic noisy data of y=-3*x^2+5*x")
        plt.savefig(filepath)
        plt.show()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )

    def denormalize_x(self, x_array: np.ndarray) -> np.ndarray:
        """If the dataset has been normalized, convert x_array back to original scale.

        x_array can be a numpy array of shape (N,1) or (N,).
        """
        # If no normalization parameters are present, return input unchanged
        if not hasattr(self, "x_mean") or not hasattr(self, "x_std"):
            return x_array
        return x_array * float(self.x_std) + float(self.x_mean)


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    dataset = NoisyRegressionDataset()
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    # save the plot
    dataset.plot(output_folder / "plot_dataset_example.png")
    # dataset.plot(output_folder / "plot_dataset_example.png")
