import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_window, output_class_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_forecast = nn.Linear(hidden_size, 34)  
        self.fc_class = nn.Linear(hidden_size, output_class_size)  
        self.forecast_window = forecast_window

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]              
        forecast = self.fc_forecast(out)        
        classify = self.fc_class(last_hidden)    
        return forecast, classify
