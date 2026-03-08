from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize


class ImageFolderDataset(Dataset):
    """Generic image folder dataset.

    Expected structure: <root>/<class_name>/*.jpg (or jpeg/png).
    Class ids are assigned by sorted class_name order, 0-based.
    """

    def __init__(
        self,
        path: str,
        img_size: tuple[int, int] = (300, 200),
        img_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        data_path = Path(path)
        assert data_path.exists(), f"Path {path} does not exist"
        assert data_path.is_dir(), f"Path {path} is not a directory"

        exts = ("*.jpg", "*.jpeg", "*.png")
        imgs: list[Path] = []
        for ext in exts:
            imgs.extend(data_path.glob(f"**/{ext}"))
        self.imgs = sorted(imgs)
        assert self.imgs, f"No image files found in {path}"

        self.img_transform = img_transform
        self.img_size = img_size

        class_names = sorted({p.parent.name for p in self.imgs})
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.imgs)

    def _get_class_name(self, file_path: Path) -> str:
        return file_path.parent.name

    def _get_class_idx(self, file_path: Path) -> int:
        return self.class_to_idx[self._get_class_name(file_path)]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.imgs[idx]

        # read_image -> uint8 tensor in CxHxW
        img = read_image(str(img_path), mode=ImageReadMode.RGB)
        img = Resize(self.img_size)(img)
        img = img.float() / 255.0

        if self.img_transform is not None:
            img = self.img_transform(img)

        img_class_t = torch.tensor(self._get_class_idx(img_path), dtype=torch.long)
        return img, img_class_t


# Backward compatibility alias
BirdImgDataset = ImageFolderDataset
