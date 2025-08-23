from torch.utils.data import Dataset
import torch

class PoseDataset(Dataset):
    def __init__(self, data_dict, return_subject=False):
        self.src = data_dict["src"]
        self.trg_forecast = data_dict["trg_forecast"]
        self.trg_class = data_dict["trg_class"]
        self.subjects = data_dict["subjects"]
        self.motion_types = data_dict["motion_types"]
        self.return_subject = return_subject

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.src[idx], dtype=torch.float32)
        forecast_tensor = torch.tensor(self.trg_forecast[idx], dtype=torch.float32)
        class_tensor = torch.tensor(self.trg_class[idx], dtype=torch.float32)

        if self.return_subject and self.subjects is not None:
            return (src_tensor, (forecast_tensor, class_tensor), self.subjects[idx], self.motion_types[idx])
        else:
            return (src_tensor, (forecast_tensor, class_tensor))
