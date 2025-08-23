import sys
import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from train import mpjpe
from utils.dataset import PoseDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lstm import LSTMModel
from models.rnn import RNNModel
from models.mlp import MLP  

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),     # Head connections
    (5, 6),                              # Shoulder to shoulder
    (5, 11), (6, 12), (11, 12),         # Torso
    (5, 7), (7, 9),                     # Left arm
    (6, 8), (8, 10),                    # Right arm  
    (11, 13), (13, 15),                 # Left leg
    (12, 14), (14, 16),                 # Right leg
    (0, 5), (0, 6)                      # Head to torso
]

def draw_pose(ax, pose, color, alpha=1.0, linewidth=2):
    for (i, j) in SKELETON_EDGES:
        if i < pose.shape[0] and j < pose.shape[0]:
            if not (np.allclose(pose[i], 0) or np.allclose(pose[j], 0)):
                ax.plot([pose[i, 0], pose[j, 0]], 
                       [-pose[i, 1], -pose[j, 1]], 
                       color=color, linewidth=linewidth, alpha=alpha)
    
    ax.scatter(pose[:, 0], -pose[:, 1], 
              c=color, s=30, alpha=alpha, 
              edgecolors='white', linewidth=0.5, zorder=10)

def visualize_single_frame(pred_batch, tgt_batch, model_name, window_idx=36, frame_idx=0):
    os.makedirs("results/plots", exist_ok=True)

    pred_pose = pred_batch[window_idx, frame_idx].reshape(17, 2)
    true_pose = tgt_batch[window_idx, frame_idx].reshape(17, 2)

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_pose(ax, true_pose, color='blue', alpha=0.7, linewidth=3)
    draw_pose(ax, pred_pose, color='red', alpha=0.9, linewidth=2)

    ax.set_aspect('equal')
    ax.axis('off')

    all_points = np.concatenate([pred_pose, true_pose], axis=0)
    valid_points = all_points[~np.all(np.isclose(all_points, 0), axis=1)]

    if len(valid_points) > 0:
        margin = 0.1
        x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
        y_min, y_max = -valid_points[:, 1].max(), -valid_points[:, 1].min()

        x_range = x_max - x_min
        y_range = y_max - y_min

        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    legend_elements = [
        mpatches.Patch(color='blue', label='Ground Truth', alpha=0.7),
        mpatches.Patch(color='red', label='Prediction', alpha=0.9)
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=10)

    plt.tight_layout()
    save_path = f"results/plots/prediction_single_{model_name}_W{window_idx}_F{frame_idx}.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"[INFO] Saved single-frame visualization to {save_path}")

def visualize_cross_windows(pred_batch, tgt_batch, model_name, batch_idx=0, 
                           start_sample=0, frame_stride=3, num_frames=8):
    os.makedirs("results", exist_ok=True)
    
    batch_size, seq_len, pose_dim = pred_batch.shape  # (32, 15, 34)
    
    selected_frames = []
    selected_windows = []
    current_sample = start_sample
    current_frame = 0
    
    frame_count = 0
    while frame_count < num_frames and current_sample < batch_size:
        if current_frame < seq_len:
            selected_windows.append(current_sample)
            selected_frames.append(current_frame)
            current_frame += frame_stride
            frame_count += 1
        else:
            # Move to next sliding window and reset frame counter
            current_sample += 1
            current_frame = current_frame - seq_len  # Continue the stride pattern
            if current_frame < 0:
                current_frame = 0
    
    print(f"[INFO] Using {len(selected_frames)} frames from windows {selected_windows} at frames {selected_frames}")
    
    fig, axs = plt.subplots(1, len(selected_frames), 
                           figsize=(len(selected_frames) * 3, 4))
    
    if len(selected_frames) == 1:
        axs = [axs]
    
    for i, (window_idx, frame_idx) in enumerate(zip(selected_windows, selected_frames)):
        ax = axs[i]
        
        pred_pose = pred_batch[window_idx, frame_idx].reshape(17, 2)
        true_pose = tgt_batch[window_idx, frame_idx].reshape(17, 2)
        
        draw_pose(ax, true_pose, color='blue', alpha=0.7, linewidth=3)
        draw_pose(ax, pred_pose, color='red', alpha=0.9, linewidth=2)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        all_points = np.concatenate([pred_pose, true_pose], axis=0)
        valid_points = all_points[~np.all(np.isclose(all_points, 0), axis=1)]
        
        if len(valid_points) > 0:
            margin = 0.1
            x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
            y_min, y_max = -valid_points[:, 1].max(), -valid_points[:, 1].min()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    
    legend_elements = [
        mpatches.Patch(color='blue', label='Ground Truth', alpha=0.7),
        mpatches.Patch(color='red', label='Prediction', alpha=0.9)
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.95))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    save_path = f"results/plots/prediction_viz_{model_name}_batch_{batch_idx}_stride_{frame_stride}.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()
    
    print(f"[INFO] Saved cross-window visualization to {save_path}")

def evaluate(model_path="results/saved_models/lstm_1000.pt", batch_size=32, visualize_motion="5_forward_falls"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(os.path.dirname(__file__), "dataset.pkl"), "rb") as f:
        raw_data = pickle.load(f)

    subjects = raw_data["subjects"]
    motion_types = raw_data["motion_types"]
    unique_subjects = sorted(set(subjects))
    test_subjects = unique_subjects[-2:]

    test_data = {
        "src": [], "trg_forecast": [], "trg_class": [],
        "subjects": [], "motion_types": []
    }

    for i, subj in enumerate(subjects):
        if subj in test_subjects:
            test_data["src"].append(raw_data["src"][i])
            test_data["trg_forecast"].append(raw_data["trg_forecast"][i])
            test_data["trg_class"].append(raw_data["trg_class"][i])
            test_data["subjects"].append(subj)
            test_data["motion_types"].append(raw_data["motion_types"][i])

    test_dataset = PoseDataset(test_data, return_subject=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    subject_motion_windows = defaultdict(Counter)

    for src, (trg_forecast, trg_class), subjects, motions in test_loader:
        for subj, motion in zip(subjects, motions):
            subject_motion_windows[subj][motion] += 1

    print("\n[INFO] Sliding window count per subject and motion type:")
    for subj in sorted(subject_motion_windows.keys()):
        print(f"{subj}:")
        for motion, count in subject_motion_windows[subj].items():
            print(f"  - {motion}: {count} sliding windows")

    input_window = len(test_data["src"][0])
    keypoints_dim = len(test_data["src"][0][0]) * len(test_data["src"][0][0][0])  # 17 * 2 = 34

    model = LSTMModel(
        input_size=34,
        hidden_size=128,
        forecast_window=input_window,
        output_class_size=2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    loss_forecast = torch.nn.MSELoss()
    loss_class = torch.nn.CrossEntropyLoss()
    total_loss_f = 0.0
    total_loss_c = 0.0
    correct, total = 0, 0
    all_mpjpe = []

    visualized = False

    with torch.no_grad():
        total_loss_f, total_loss_c = 0.0, 0.0
        correct, total = 0, 0
        all_mpjpe = []
        visualized = False
        motion_forecast = []
        motion_target = []

        for idx, batch in enumerate(test_loader):
            if len(batch) == 4:
                src, (trg_forecast, trg_class), subjects, motions = batch
            else:
                src, (trg_forecast, trg_class) = batch
                print("batch size only has 2 elements")
                continue

            src = src.to(device)
            trg_forecast = trg_forecast.to(device)
            trg_class = trg_class.to(device)

            src = src.view(src.size(0), src.size(1), -1)
            trg_forecast = trg_forecast.view(trg_forecast.size(0), trg_forecast.size(1), -1)

            forecast_out, class_out = model(src)

            loss_f = loss_forecast(forecast_out, trg_forecast)
            loss_c = loss_class(class_out, torch.argmax(trg_class, dim=1))

            total_loss_f += loss_f.item() * src.size(0)
            total_loss_c += loss_c.item() * src.size(0)

            pred = torch.argmax(class_out, dim=1)
            correct += (pred == torch.argmax(trg_class, dim=1)).sum().item()
            total += trg_class.size(0)

            all_mpjpe.append(mpjpe(forecast_out, trg_forecast).item())

            for i in range(len(motions)):
                motion_str = motions[i]
                if isinstance(motion_str, bytes):
                    motion_str = motion_str.decode('utf-8')

                if motion_str == visualize_motion:
                    motion_forecast.append(forecast_out[i:i+1].cpu())  # shape: (1, seq_len, dim)
                    motion_target.append(trg_forecast[i:i+1].cpu())

        if motion_forecast and not visualized:
            pred_all = torch.cat(motion_forecast, dim=0).numpy()
            target_all = torch.cat(motion_target, dim=0).numpy()

            print(f"[INFO] Visualizing full motion: {visualize_motion} with {pred_all.shape[0]} windows")
            visualize_cross_windows(
                pred_all,
                target_all,
                model_name="mlp",
                batch_idx=0,           
                start_sample=0,
                frame_stride=135,        
                num_frames=8          
            )
            visualized = True

        # visualize only W36 F0
        if pred_all.shape[0] > 36:
            visualize_single_frame(
                pred_all,
                target_all,
                model_name="mlp",
                window_idx=36,
                frame_idx=0
            )
        else:
            print("[WARNING] Not enough windows to extract window 36.")

        avg_loss_f = total_loss_f / total
        avg_loss_c = total_loss_c / total
        avg_mpjpe = sum(all_mpjpe) / len(all_mpjpe)
        acc = correct / total

        print(f"[Evaluation] Forecast Loss: {avg_loss_f:.4f} | Class Loss: {avg_loss_c:.4f} | "
            f"MPJPE: {avg_mpjpe:.4f} | Accuracy: {acc:.4f}")
        
if __name__ == "__main__":
    evaluate()