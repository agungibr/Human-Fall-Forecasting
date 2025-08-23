import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_window, output_class_size):
        super().__init__()
        self.input_size = input_size
        self.forecast_window = forecast_window
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.forecast_head = nn.Linear(hidden_size, forecast_window * 34)
        self.class_head = nn.Linear(hidden_size, output_class_size)

    def forward(self, x):
        batch_size = x.size(0)
        shared = self.shared(x)
        forecast = self.forecast_head(shared).view(batch_size, self.forecast_window, 34)
        classify = self.class_head(shared)
        return forecast, classify
