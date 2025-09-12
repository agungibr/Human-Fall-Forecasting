import torch
import torch.nn as nn

class TinyTransformerModel(nn.Module):
    def __init__(self, input_size, forecast_window, output_class_size,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)  

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_forecast = nn.Linear(d_model, input_size)
        self.fc_class = nn.Linear(d_model, output_class_size)
        self.forecast_window = forecast_window

    def forward(self, x):
        x = self.input_proj(x)  
        out = self.transformer(x)  
        forecast = self.fc_forecast(out)  
        last_hidden = out[:, -1, :]      
        classify = self.fc_class(last_hidden) 
        return forecast, classify
