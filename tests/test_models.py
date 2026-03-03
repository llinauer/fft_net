import pytest
import torch

from fft_net.model import FFTCNN, FFTNet


def test_fftnet_forward_shape() -> None:
    model = FFTNet(img_size=(128, 128), num_classes=5, hidden_dims=(64, 32), dropout=0.1)
    x = torch.rand(4, 3, 128, 128)
    y = model(x)
    assert y.shape == (4, 5)


def test_fftcnn_forward_shape() -> None:
    model = FFTCNN(num_classes=7, conv_channels=(8, 16), dropout=0.1)
    x = torch.rand(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 7)


@pytest.mark.parametrize("bad_channels", [(), (0, 8), (-1, 8, 16)])
def test_fftcnn_invalid_channels_raises(bad_channels: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        FFTCNN(num_classes=3, conv_channels=bad_channels)
