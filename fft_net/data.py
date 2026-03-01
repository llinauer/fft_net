from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize


class BirdImgDataset(Dataset):
    def __init__(
        self,
        path: str,
        img_size: tuple[int, int] = (300, 200),
        img_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        data_path = Path(path)
        assert data_path.exists(), f"Path {path} does not exist"
        assert data_path.is_dir(), f"Path {path} is not a directory"

        self.imgs = sorted(data_path.glob("**/*.jpg"))
        assert self.imgs, f"No .jpg files found in {path}"

        self.img_transform = img_transform
        self.img_size = img_size

        # Validate CUB-style labels and normalize to 0-based in __getitem__.
        labels_1_based = [self._get_class(p) for p in self.imgs]
        min_label = min(labels_1_based)
        max_label = max(labels_1_based)
        if min_label != 1:
            raise ValueError(f"Expected 1-based labels (min=1), got min={min_label}")
        expected = set(range(1, max_label + 1))
        found = set(labels_1_based)
        if found != expected:
            missing = sorted(expected - found)[:10]
            raise ValueError(f"Expected contiguous labels 1..N, missing={missing}")

        self.num_classes = max_label

    def __len__(self) -> int:
        return len(self.imgs)

    @staticmethod
    def _get_class(file_path: Path) -> int:
        dir_name = file_path.parent.name
        class_idx, _class_name = dir_name.split(".", maxsplit=1)
        return int(class_idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.imgs[idx]

        # read_image -> uint8 tensor in CxHxW
        img = read_image(str(img_path), mode=ImageReadMode.RGB)
        img = Resize(self.img_size)(img)
        img = img.float() / 255.0

        if self.img_transform is not None:
            img = self.img_transform(img)

        # normalize class ids from 1..N to 0..(N-1)
        img_class_t = torch.tensor(self._get_class(img_path) - 1, dtype=torch.long)
        return img, img_class_t
