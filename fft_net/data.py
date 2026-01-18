import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataSet
from torchvision.transform import Transform, Resize, ToTensor

class BirdImgDataset(DataSet):

    def __init__(self, path: str, img_size: tuple[int, int] = (300, 200), img_transform: Transform | None = None):

        data_path = Path(path)
        assert data_path.exists(), f"Path {path} does not exist"
        assert data_path.is_dir() , f"Path {path} is not a directory"
        self.imgs = data_path.glob("**/*.jpg")
        self.img_transform = img_transform
        self.img_size

    def __len__(self):
        return len(list(self.imgs))
    
    @classmethod
    def _get_class(file_path: str) -> tuple[int, str]:
        dir_name = file_path.parent.name
        class_idx, class_name = dir_name.split(".")
        return int(class_idx), class_name

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.LongTensor]:
        img = torchvision.io.decode(self.imgs[idx])
        img = Resize(self.img_size)(img)

        if self.img_transform is not None:
            img_t = self.img_transform(img_t)
        img_t = ToTensor()(img)

        img_class_t = torch.LongTensor(self._get_class(self.imgs[idx]))
        return img_t, img_class_t
