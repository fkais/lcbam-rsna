import torch
import torch.nn as nn


class LCBAMv2(nn.Module):
    """
    Lightweight CBAM v2.

    Structure:
        Input
          -> Lightweight Channel Attention
          -> Dilated Spatial Attention
          -> Residual Fusion

    Args:
        kernel_size: Kernel size of the 1D channel interaction.
        dilation: Dilation used in spatial attention.
        gamma_init: Initial residual scaling factor.
        spatial_kernel: Kernel size of the dilated spatial convolution.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        dilation: int = 3,
        gamma_init: float = 0.0,
        spatial_kernel: int = 3,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        if dilation < 1:
            raise ValueError("dilation must be >= 1.")

        if spatial_kernel < 1 or spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be a positive odd integer.")

        # -------- Channel Attention --------
        #
        # Input descriptor shape:
        # [B, 2, C]
        #
        # AvgPool and MaxPool are treated as 2 input channels.
        # Conv1d performs local interaction along channel dimension C.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        # -------- Spatial Attention --------
        #
        # The default 3x3 with dilation=3 gives an effective 7x7 receptive field.
        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel,
            padding=dilation * (spatial_kernel // 2),
            dilation=dilation,
            bias=False,
        )

        # -------- Residual Fusion --------
        #
        # Start from identity mapping for more stable optimization.
        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init))
        )

        self.sigmoid = nn.Sigmoid()

    def channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute channel attention.

        Args:
            x: [B, C, H, W]

        Returns:
            attention: [B, C, 1, 1]
        """

        avg_desc = self.avg_pool(x).flatten(1)
        max_desc = self.max_pool(x).flatten(1)

        # [B, 2, C]
        descriptor = torch.stack(
            [avg_desc, max_desc],
            dim=1,
        )

        # [B, 1, C]
        attention = self.channel_conv(descriptor)

        # [B, C, 1, 1]
        attention = (
            self.sigmoid(attention)
            .squeeze(1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        return attention

    def spatial_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention.

        Args:
            x: [B, C, H, W]

        Returns:
            attention: [B, 1, H, W]
        """

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True,
        )

        # [B, 2, H, W]
        spatial_descriptor = torch.cat(
            [avg_map, max_map],
            dim=1,
        )

        attention = self.spatial_conv(
            spatial_descriptor
        )

        return self.sigmoid(attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Channel attention
        channel_weight = self.channel_attention(x)
        x = x * channel_weight

        # Spatial attention
        spatial_weight = self.spatial_attention(x)
        attended = x * spatial_weight

        # Residual fusion
        out = identity + self.gamma * attended

        return out
