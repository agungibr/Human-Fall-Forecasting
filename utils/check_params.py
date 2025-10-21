import os
import sys
import pickle
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model definitions
from models.mlp import MLP
from models.lstm import LSTMModel
from models.rnn import RNNModel
from models.gru import GRUModel
from models.stgcn import STGCN
from models.tiny_transformer import TinyTransformerModel

def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_dataset(pkl_path):
    """Loads and inspects the preprocessed dataset."""
    print(f"[INFO] Loading dataset from {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("\n[INFO] Dataset keys and shapes:")
    for k, v in data.items():
        # Using a check for lists since 'subjects' and 'motion_types' are lists of strings
        if isinstance(v, list):
             print(f"  - {k}: length {len(v)}")
        else:
             print(f"  - {k}: {np.array(v).shape}")
    return data

def inspect_models(input_window, feature_dim, forecast_window, num_classes, num_coords):
    """Initializes all models and prints their parameter counts."""
    mlp_input_size = input_window * feature_dim
    recurrent_input_size = feature_dim

    print("\n[INFO] Initializing models and counting parameters\n" + "="*50)

    # MLP
    mlp = MLP(input_size=mlp_input_size, hidden_size=128, forecast_window=forecast_window, output_class_size=num_classes)
    print(f"[MLP]\n  - Input Size (flattened): {mlp_input_size}\n  - Trainable Params: {count_parameters(mlp):,}")

    # LSTM
    lstm = LSTMModel(input_size=recurrent_input_size, hidden_size=128, forecast_window=forecast_window, output_class_size=num_classes)
    print(f"\n[LSTM]\n  - Input Size (per frame): {recurrent_input_size}\n  - Trainable Params: {count_parameters(lstm):,}")

    # RNN
    rnn = RNNModel(input_size=recurrent_input_size, hidden_size=128, forecast_window=forecast_window, output_class_size=num_classes)
    print(f"\n[RNN]\n  - Input Size (per frame): {recurrent_input_size}\n  - Trainable Params: {count_parameters(rnn):,}")

    # GRU
    gru = GRUModel(input_size=recurrent_input_size, hidden_size=128, forecast_window=forecast_window, output_class_size=num_classes)
    print(f"\n[GRU]\n  - Input Size (per frame): {recurrent_input_size}\n  - Trainable Params: {count_parameters(gru):,}")

    # Transformer
    transformer = TinyTransformerModel(input_size=recurrent_input_size, forecast_window=forecast_window, output_class_size=num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1)
    print(f"\n[TRANSFORMER]\n  - Input Size (per frame): {recurrent_input_size}\n  - Trainable Params: {count_parameters(transformer):,}")

    # STGCN
    # FIX 2: Define graph_args where it's needed
    graph_args = {'layout': 'coco', 'strategy': 'spatial'}
    stgcn = STGCN(
        in_channels=num_coords,
        num_class=num_classes,
        graph_args=graph_args,
        # FIX 3: Use the 'forecast_window' variable that was passed into the function
        forecast_window=forecast_window,
        edge_importance_weighting=True
    )
    print(f"\n[STGCN]\n  - Input Channels (C): {num_coords}\n  - Trainable Params: {count_parameters(stgcn):,}")
    print("="*50)

if __name__ == "__main__":
    # Correctly locate dataset.pkl in the project root, not inside utils/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "dataset.pkl")

    data = load_dataset(dataset_path)

    # Dynamically determine dimensions from the loaded data
    src = np.array(data["src"])
    trg = np.array(data["trg_forecast"])

    input_window = src.shape[1]
    num_joints = src.shape[2]
    num_coords = src.shape[3]
    feature_dim = num_joints * num_coords

    forecast_window = trg.shape[1]
    num_classes = 2 # Fall vs Non-Fall

    print("\n[INFO] Data Dimensions:")
    print(f"  - Input Window (T_in): {input_window} frames")
    print(f"  - Forecast Window (T_out): {forecast_window} frames")
    print(f"  - Feature Dimension (per frame): {feature_dim} ( {num_joints} joints × {num_coords} coords )")
    
    # FIX 4: Pass the required num_coords variable to the function
    inspect_models(input_window, feature_dim, forecast_window, num_classes, num_coords)