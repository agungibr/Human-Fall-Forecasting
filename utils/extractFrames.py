import cv2
import os

class extractFrames:
    def __init__(self, dataset_path="../../dataset", output_path="../../frames"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    def extract_frames(self):
        categories = ["fall", "non-fall"]
        for category in categories:
            category_path = os.path.join(self.dataset_path, category)
            subjects = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]

            for subject in subjects:
                subject_path = os.path.join(category_path, subject)
                videos = [f for f in os.listdir(subject_path) if f.endswith(".mp4")]

                for video in videos:
                    video_path = os.path.join(subject_path, video)
                    self._extract_video_frames(video_path, category, subject, video)

    def _extract_video_frames(self, video_path, category, subject, video_name):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return

        # Prepare output folder
        name_without_ext = os.path.splitext(video_name)[0]
        output_dir = os.path.join(self.output_path, category, subject, name_without_ext)
        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {video_path} to {output_dir}")

if __name__ == "__main__":
    extractor = extractFrames()
    extractor.extract_frames()
