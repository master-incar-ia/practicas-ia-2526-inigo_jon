import torch
import torch.nn as nn

# LSTM para señales 1D (ECG)
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=2, output_dim=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: (batch, seq_len) o (batch, seq_len, 1)
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # último hidden state
        out = self.fc(out)
        return out

if __name__ == "__main__":
    pass