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
