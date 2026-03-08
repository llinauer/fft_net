from pathlib import Path

import torch
from torchvision.io import write_jpeg

from fft_net.data import ImageFolderDataset


def _write_dummy_jpg(path: Path, height: int = 48, width: int = 64) -> None:
    img = torch.randint(0, 256, (3, height, width), dtype=torch.uint8)
    write_jpeg(img, str(path))


def test_bird_dataset_normalizes_labels_and_loads_images(tmp_path: Path) -> None:
    class_a = tmp_path / "001.class_a"
    class_b = tmp_path / "002.class_b"
    class_a.mkdir(parents=True)
    class_b.mkdir(parents=True)

    _write_dummy_jpg(class_a / "a1.jpg")
    _write_dummy_jpg(class_a / "a2.jpg")
    _write_dummy_jpg(class_b / "b1.jpg")

    ds = ImageFolderDataset(path=str(tmp_path), img_size=(32, 32))

    assert len(ds) == 3
    assert ds.num_classes == 2

    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.float32
    assert 0.0 <= float(x.min()) <= 1.0
    assert 0.0 <= float(x.max()) <= 1.0
    assert int(y) in {0, 1}
