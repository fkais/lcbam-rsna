import torch
import torch.nn as nn


class LCBAM(nn.Module):
    """
    Legacy LCBAM implementation used in the original experiments.

    NOTE:
    This implementation is retained only for reproducibility of
    historical experiments. It is essentially a lazily initialized
    CBAM-style attention module and is NOT the new LCBAMv2.
    """

    def __init__(self, c=None, reduction=16):
        super().__init__()
        self.c = c
        self.reduction = reduction
        self.initialized = False

    def _build(self, c):
        # Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(c, c // self.reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // self.reduction, c, 1, bias=False),
        )

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

        self.initialized = True

    def forward(self, x):
        # Lazily construct module according to input channels
        if not self.initialized:
            c = x.shape[1]
            self._build(c)

        # Channel attention
        avg_out = self.mlp(self.avgpool(x))
        max_out = self.mlp(self.maxpool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(
            self.conv_spatial(
                torch.cat([avg_out, max_out], dim=1)
            )
        )
        x = x * sa

        return x