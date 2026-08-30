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