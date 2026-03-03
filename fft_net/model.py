import torch
from torch import nn


class FFTNet(nn.Module):
    """Simple classifier that uses FFT magnitude features + MLP."""

    def __init__(
        self,
        img_size: tuple[int, int],
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        flat_dim = img_size[0] * img_size[1]

        layers: list[nn.Module] = []
        in_dim = flat_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.mean(dim=1)  # -> [B, H, W] grayscale
        x = torch.fft.fft2(x)
        x = torch.log1p(torch.abs(x))
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class FFTCNN(nn.Module):
    """CNN classifier applied to the 2D FFT magnitude of the input image."""

    def __init__(
        self,
        num_classes: int,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if len(conv_channels) < 1:
            raise ValueError("FFTCNN expects at least one conv channel value")
        if any(ch <= 0 for ch in conv_channels):
            raise ValueError(f"All conv channel values must be > 0, got: {conv_channels}")

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(conv_channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.mean(dim=1)  # -> [B, H, W]
        x = torch.fft.fft2(x)  # complex64/complex128
        x = torch.log1p(torch.abs(x))  # -> [B, H, W] real magnitude map
        x = x.unsqueeze(1)  # -> [B, 1, H, W]

        x = self.features(x)
        return self.classifier(x)
