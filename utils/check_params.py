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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_dataset(pkl_path):
    print(f"[INFO] Loading dataset from {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("\n[INFO] Dataset keys and shapes:")
    for k, v in data.items():
        print(f"  - {k}: {np.array(v).shape}")
    return data

def inspect_models(input_window, feature_dim, forecast_window, num_classes):
    mlp_input_size = input_window * feature_dim   # 15 * 34 = 510
    rnn_input_size = feature_dim                  # 34 (per frame)

    print("\n[INFO] Initializing models...\n")

    # MLP
    mlp = MLP(
        input_size=mlp_input_size,
        hidden_size=128,
        forecast_window=forecast_window,
        output_class_size=num_classes
    )
    print("[MLP] Input Size =", mlp_input_size)
    print("[MLP] Total trainable params:", count_parameters(mlp))

    # LSTM
    lstm = LSTMModel(
        input_size=rnn_input_size,
        hidden_size=128,
        forecast_window=forecast_window,
        output_class_size=num_classes
    )
    print("\n[LSTM] Input Size =", rnn_input_size)
    print("[LSTM] Total trainable params:", count_parameters(lstm))

    # RNN
    rnn = RNNModel(
        input_size=rnn_input_size,
        hidden_size=128,
        forecast_window=forecast_window,
        output_class_size=num_classes
    )
    print("\n[RNN] Input Size =", rnn_input_size)
    print("[RNN] Total trainable params:", count_parameters(rnn))

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.pkl")

    data = load_dataset(dataset_path)

    src = np.array(data["src"])  # (N, 15, 17, 2)
    input_window = src.shape[1]         # 15 frames
    num_joints = src.shape[2]           # 17
    coord_dims = src.shape[3]           # 2 (x, y)
    keypoints_dim = num_joints * coord_dims  # 17 × 2 = 34

    forecast_window = input_window
    num_classes = 2

    print(input_window)
    print(keypoints_dim)
    
    inspect_models(input_window, keypoints_dim, forecast_window, num_classes)
