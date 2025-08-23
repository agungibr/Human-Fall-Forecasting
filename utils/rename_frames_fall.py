import os

fall_start_info = {
    "subject-1": {
        "1_backward_falls": 114,
        "1_forward_falls": 90,
        "1_left_falls": 105,
        "1_right_falls": 101,
        "1_sitting_falls": 202,
        "1_standing_falls": 73
    },
    "subject-2": {
        "2_backward_falls": 71,
        "2_forward_falls": 64,
        "2_left_falls": 99,
        "2_right_falls": 108,
        "2_sitting_falls": 200,
        "2_standing_falls": 64
    },
    "subject-3": {
        "3_backward_falls": 143,
        "3_forward_falls": 149,
        "3_left_falls": 186,
        "3_right_falls": 141,
        "3_sitting_falls": 420,
        "3_standing_falls": 114
    },
    "subject-4": {
        "4_backward_falls": 159,
        "4_forward_falls": 164,
        "4_left_falls": 187,
        "4_right_falls": 155,
        "4_sitting_falls": 367,
        "4_standing_falls": 177
    },
    "subject-5": {
        "5_backward_falls": 124,
        "5_forward_falls": 120,
        "5_left_falls": 118,
        "5_right_falls": 124,
        "5_sitting_falls": 360,
        "5_standing_falls": 123
    },
    "subject-6": {
        "6_backward_falls": 171,
        "6_forward_falls": 130,
        "6_left_falls": 189,
        "6_right_falls": 160,
        "6_sitting_falls": 396,
        "6_standing_falls": 140
    }
}

def rename_frames_with_labels(folder_path, fall_start):
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder '{folder_path}' tidak ditemukan.")
        return

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    renamed_count = 0

    for f in files:
        try:
            frame_num = int(f.split("_")[-1].split(".")[0])
        except ValueError:
            print(f"[SKIP] Format file tidak sesuai: {f}")
            continue

        label = 1 if frame_num >= fall_start else 0
        frame_base = f.split("_")[-1]
        new_name = f"{label}_frame_{frame_base}"

        src = os.path.join(folder_path, f)
        dst = os.path.join(folder_path, new_name)

        if src != dst:
            os.rename(src, dst)
            renamed_count += 1

    print(f"[DONE] {renamed_count} frame dilabeli di '{folder_path}'.")

def process_all_fall_videos(base_folder="../../frames/fall"):
    for subject, videos in fall_start_info.items():
        for video_name, fall_start in videos.items():
            video_path = os.path.join(base_folder, subject, video_name)
            rename_frames_with_labels(video_path, fall_start)

process_all_fall_videos()
