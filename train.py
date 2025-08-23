import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import MLP
from models.rnn import RNNModel
from models.lstm import LSTMModel
from sklearn.metrics import confusion_matrix, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# MPJPE Metric
# -------------------------
def mpjpe(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

model_name = "lstm"  
batch_size = 32
num_epochs = 10
lr = 1e-4

# -------------------------
# Training Loop
# -------------------------
def train(model, train_loader, num_epochs=10, lr=1e-4, model_name="default_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_forecast = nn.MSELoss()
    loss_class = nn.CrossEntropyLoss()

    epoch_losses = []
    start_time = time.time()
    all_preds, all_labels = [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_loss_f, total_loss_c = 0.0, 0.0, 0.0
        correct, total = 0, 0
        all_mpjpe = []

        y_true_epoch, y_pred_epoch = [], []

        for src, (trg_forecast, trg_class), _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            src = src.to(device)
            trg_forecast = trg_forecast.to(device)
            trg_class = trg_class.to(device)

            src = src.view(src.size(0), src.size(1), -1)
            trg_forecast = trg_forecast.view(trg_forecast.size(0), trg_forecast.size(1), -1)

            optimizer.zero_grad()
            forecast_out, class_out = model(src)

            loss_f = loss_forecast(forecast_out, trg_forecast)
            loss_c = loss_class(class_out, torch.argmax(trg_class, dim=1))
            loss = loss_f + loss_c

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * src.size(0)
            total_loss_f += loss_f.item() * src.size(0)
            total_loss_c += loss_c.item() * src.size(0)

            all_mpjpe.append(mpjpe(forecast_out, trg_forecast).item())
            preds = torch.argmax(class_out, dim=1)
            labels = torch.argmax(trg_class, dim=1)
            correct += (preds == torch.argmax(trg_class, dim=1)).sum().item()
            total += trg_class.size(0)

            y_pred_epoch.extend(preds.cpu().numpy())
            y_true_epoch.extend(labels.cpu().numpy())

        all_preds.extend(y_pred_epoch)
        all_labels.extend(y_true_epoch)

        avg_loss = total_loss / total
        avg_loss_f = total_loss_f / total
        avg_loss_c = total_loss_c / total
        avg_mpjpe = sum(all_mpjpe) / len(all_mpjpe)
        acc = correct / total

        epoch_losses.append(avg_loss)

        print(f"[Epoch {epoch+1}] Train → Loss: {avg_loss:.4f} | Forecast: {avg_loss_f:.4f} | "
              f"Class: {avg_loss_c:.4f} | MPJPE: {avg_mpjpe:.4f} | Acc: {acc:.4f}")
        
        cm = confusion_matrix(y_true_epoch, y_pred_epoch)
        print(f"[Epoch {epoch+1}] Confusion Matrix:\n{cm}")

    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f"[INFO] Total Training Time: {total_minutes:.2f} minutes")

    # Save the model to 
    model_path = f"results/saved_models/{model_name}_{num_epochs}.pt"
    torch.save(model.state_dict(), model_path)

    print(f"[INFO] Model saved to {model_path}")

    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, color='blue', linewidth=2)
    plt.title("Training Loss Over Epochs (RNN)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = f"results/plots/{model_name}_loss_curve_{num_epochs}.png"
    plt.savefig(loss_plot_path, dpi=600)
    print(f"[INFO] Loss curve saved to {loss_plot_path}")

# -------------------------
# Run Training
# -------------------------
if __name__ == "__main__":
    import os
    import pickle
    from utils.dataset import PoseDataset
    from torch.utils.data import DataLoader
    from collections import defaultdict, Counter

    with open(os.path.join(os.path.dirname(__file__), "dataset.pkl"), "rb") as f:
        raw_data = pickle.load(f)

    subjects = raw_data["subjects"]
    motion_types = raw_data["motion_types"]
    unique_subjects = sorted(set(subjects))
    train_subjects = unique_subjects[0:4]

    # Prepare train_data dict
    train_data = {
        "src": [], "trg_forecast": [], "trg_class": [],
        "subjects": [], "motion_types": []
    }
    
    for i, subj in enumerate(subjects):
        if subj in train_subjects:
            train_data["src"].append(raw_data["src"][i])
            train_data["trg_forecast"].append(raw_data["trg_forecast"][i])
            train_data["trg_class"].append(raw_data["trg_class"][i])
            train_data["subjects"].append(subj)
            train_data["motion_types"].append(raw_data["motion_types"][i])

    train_dataset = PoseDataset(train_data, return_subject=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    subject_to_motions = defaultdict(set)

    subject_motion_windows = defaultdict(Counter)

    for src, (trg_forecast, trg_class), subjects, motions in train_loader:
        for subj, motion in zip(subjects, motions):
            subject_motion_windows[subj][motion] += 1

    print("\n[INFO] Sliding window count per subject and motion type:")
    for subj in sorted(subject_motion_windows.keys()):
        print(f"{subj}:")
        for motion, count in subject_motion_windows[subj].items():
            print(f"  - {motion}: {count} sliding windows")

    input_window = len(train_data["src"][0])  # e.g. 15
    keypoints_dim = len(train_data["src"][0][0]) * len(train_data["src"][0][0][0])  # 17 * 2 = 34

    if model_name == "mlp":
        from models.mlp import MLP
        model = MLP(
            input_size=input_window * keypoints_dim,
            hidden_size=128,
            forecast_window=input_window,
            output_class_size=2
        )
    
    elif model_name == "rnn":
        from models.rnn import RNNModel
        model = RNNModel(
            input_size=34,
            hidden_size=128,
            forecast_window=input_window,
            output_class_size=2
        )

    elif model_name == "lstm":
        from models.lstm import LSTMModel
        model = LSTMModel(
            input_size=34,
            hidden_size=128,
            forecast_window=input_window,
            output_class_size=2
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)

    print(f"Using model: {model_name.upper()}")

    # Train and save model
    train(model, train_loader, num_epochs=num_epochs, lr=lr, model_name=model_name)