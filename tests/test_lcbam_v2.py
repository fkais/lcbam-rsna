import pytest
import torch

from src.models.lcbam_v2 import LCBAMv2


def test_output_shape():
    model = LCBAMv2()

    x = torch.randn(
        2,
        64,
        32,
        32,
    )

    y = model(x)

    assert y.shape == x.shape


def test_different_channels():
    model = LCBAMv2()

    for channels in [16, 32, 64, 128, 256]:
        x = torch.randn(
            1,
            channels,
            16,
            16,
        )

        y = model(x)

        assert y.shape == x.shape


def test_parameter_count():
    model = LCBAMv2(
        kernel_size=5,
        dilation=3,
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    # Channel Conv1d:
    # 1 * 2 * 5 = 10
    #
    # Spatial Conv2d:
    # 1 * 2 * 3 * 3 = 18
    #
    # gamma:
    # 1
    #
    # Total = 29
    assert params == 29


def test_backward():
    model = LCBAMv2(
        gamma_init=0.1
    )

    x = torch.randn(
        2,
        64,
        32,
        32,
        requires_grad=True,
    )

    y = model(x)

    loss = y.mean()
    loss.backward()

    assert x.grad is not None

    for param in model.parameters():
        assert param.grad is not None


@pytest.mark.parametrize("channel_kernel", [3, 5, 7])
@pytest.mark.parametrize("spatial_kernel", [3, 5, 7])
def test_searchable_kernels_preserve_shape(channel_kernel, spatial_kernel):
    model = LCBAMv2(
        kernel_size=channel_kernel,
        dilation=3,
        gamma_init=0.0,
        spatial_kernel=spatial_kernel,
    )

    x = torch.randn(1, 32, 16, 16)
    y = model(x)

    assert y.shape == x.shape
    assert model.channel_conv.kernel_size == (channel_kernel,)
    assert model.spatial_conv.kernel_size == (spatial_kernel, spatial_kernel)
    assert model.spatial_conv.padding == (
        3 * (spatial_kernel // 2),
        3 * (spatial_kernel // 2),
    )


@pytest.mark.parametrize("spatial_kernel", [0, 2, 4, -1])
def test_spatial_kernel_must_be_positive_and_odd(spatial_kernel):
    with pytest.raises(ValueError, match="spatial_kernel"):
        LCBAMv2(spatial_kernel=spatial_kernel)
