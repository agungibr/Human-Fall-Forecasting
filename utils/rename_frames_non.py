import os

def label_all_fall_frames_zero(base_folder="../../frames/non-fall"):
    if not os.path.isdir(base_folder):
        print(f"[ERROR] Folder '{base_folder}' tidak ditemukan.")
        return

    subject_folders = [os.path.join(base_folder, subj) for subj in os.listdir(base_folder)
                       if os.path.isdir(os.path.join(base_folder, subj))]

    for subject in subject_folders:
        video_folders = [os.path.join(subject, vid) for vid in os.listdir(subject)
                         if os.path.isdir(os.path.join(subject, vid))]

        for video_folder in video_folders:
            rename_frames_to_zero(video_folder)

def rename_frames_to_zero(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    renamed_count = 0

    for f in files:
        try:
            frame_num = int(f.split("_")[-1].split(".")[0])
        except ValueError:
            print(f"[SKIP] Format file tidak sesuai: {f}")
            continue

        frame_base = f.split("_")[-1]
        new_name = f"0_frame_{frame_base}"

        src = os.path.join(folder_path, f)
        dst = os.path.join(folder_path, new_name)

        if src != dst:
            os.rename(src, dst)
            renamed_count += 1

    print(f"[INFO] {renamed_count} frame dilabeli 0 di: {folder_path}")


# ✅ Jalankan fungsi utama
label_all_fall_frames_zero()
