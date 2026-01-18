import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataSet, DataLoader

class BirdImgDataset(DataSet):

    def __init__(self, path: str):

        data_path = Path(path)
        assert data_path.exists(), f"Path {path} does not exist"
        assert data_path.is_dir() , f"Path {path} is not a directory"
        self.imgs = data_path.glob("**/*.jpg")

    def __len__(self):
        return len(list(self.imgs))
    
    @classmethod
    def _get_class(file_path: str) -> tuple[int, str]:
        dir_name = file_path.parent.name
        class_idx, class_name = dir_name.split(".")
        return int(class_idx), class_name

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_t = torchvision.io.decode(self.imgs[idx])
