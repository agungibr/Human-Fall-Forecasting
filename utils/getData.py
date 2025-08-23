import numpy as np
import cv2 as cv
import torch
import os
import pickle
from ultralytics import YOLO
from torch.utils.data import Dataset

class getData(Dataset):
    def __init__(self, folder="../../frames/", input_window=15, output_window=15, step=5,
                 pkl=True, pkl_path="../dataset.pkl"):
        self.src = []
        self.trg_forecast = []
        self.trg_class = []
        self.subjects = []
        self.motion_types = []
        to_onehot = np.eye(2)

        if pkl and os.path.exists(pkl_path):
            print(f"[INFO] Loading preprocessed data from {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.src = data['src']
                self.trg_forecast = data['trg_forecast']
                self.trg_class = data['trg_class']
                self.subjects = data['subjects']
                self.motion_types = data['motion_types']
            return

        print("[INFO] Extracting keypoints with YOLO...")
        model = YOLO('../models/yolo11n-pose.pt')

        for subject_type in os.listdir(folder):  # fall / non-fall (ignored)
            subject_path = os.path.join(folder, subject_type)
            if not os.path.isdir(subject_path):
                continue

            for subject in sorted(os.listdir(subject_path)):  # subject-1, subject-2, ...
                subject_dir = os.path.join(subject_path, subject)
                if not os.path.isdir(subject_dir):
                    continue

                for motion in os.listdir(subject_dir):  # 1_backward_falls, 1_walking, etc.
                    motion_path = os.path.join(subject_dir, motion)
                    if not os.path.isdir(motion_path):  
                        continue

                    frame_files = sorted([
                        f for f in os.listdir(motion_path) if f.endswith(".jpg")
                    ])

                    frames = []
                    labels = []

                    for frame_file in frame_files:
                        label = int(frame_file.split("_")[0])  # 0_frame_*.jpg or 1_frame_*.jpg
                        img_path = os.path.join(motion_path, frame_file)
                        frame = cv.imread(img_path)

                        if frame is not None:
                            frame = cv.resize(frame, (500, 500))
                            frames.append(frame)
                            labels.append(label)

                    keypoints_in_video = []
                    failed_count = 0

                    for frame in frames:
                        keypoints = []
                        output = model(frame, save=False, verbose=False)
                        if output and output[0].keypoints is not None and len(output[0].keypoints) > 0:
                            kpt_data = output[0].keypoints.data[0]  # (17, 3)
                            for kp in kpt_data:
                                x, y = kp[0].item(), kp[1].item()
                                keypoints.append([x / 500, y / 500])
                        else:
                            failed_count += 1
                            keypoints = [[0, 0]] * 17  # fill with zeros

                        keypoints_in_video.append(keypoints)

                    print(f"[INFO] {subject_type}/{subject}/{motion} → Failed frames: {failed_count}")

                    subject_key = f"{subject}"

                    # Sliding window with labels
                    for slider in range(0, len(keypoints_in_video) - input_window - output_window + 1, step):
                        src_seq = keypoints_in_video[slider:slider + input_window]
                        trg_seq = keypoints_in_video[slider + input_window:slider + input_window + output_window]
                        class_seq = labels[slider + input_window:slider + input_window + output_window]

                        majority_label = int(np.round(np.mean(class_seq)))

                        self.src.append(src_seq)
                        self.trg_forecast.append(trg_seq)
                        self.trg_class.append(to_onehot[majority_label])
                        self.subjects.append(subject_key)
                        self.motion_types.append(motion)

        if pkl:
            print(f"[INFO] Saving extracted pose data to {pkl_path}...")
            with open(pkl_path, 'wb') as f:
                pickle.dump({
                    'src': self.src,
                    'trg_forecast': self.trg_forecast,
                    'trg_class': self.trg_class,
                    'subjects': self.subjects,
                    'motion_types': self.motion_types
                }, f)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src[idx], dtype=torch.float32),
            (
                torch.tensor(self.trg_forecast[idx], dtype=torch.float32),
                torch.tensor(self.trg_class[idx], dtype=torch.float32)
            )
        )

if __name__ == "__main__":
    dataset = getData()
    print(f"Loaded dataset with {len(dataset)} samples")

    from collections import Counter
    motion_counts = Counter(dataset.motion_types)
    for motion, count in sorted(motion_counts.items()):
        print(f"{motion}: {count}")