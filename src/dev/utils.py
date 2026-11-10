import torch


def setup_device() -> tuple[str, int | str, str, bool]:
    # NOTE: the 4th flag (use_compile) is intentionally False — the project never actually
    # calls torch.compile(). Wiring BERT into torch.compile would trigger constant recompiles
    # with padded dynamic seq lengths and would slow down startup.
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        use_compile = False
    else:
        accelerator = "cpu"
        devices = "auto"
        precision = "32-true"
        use_compile = False

    return (accelerator, devices, precision, use_compile)
