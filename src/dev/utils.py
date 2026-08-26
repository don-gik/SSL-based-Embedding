import torch


def setup_device() -> tuple[str, int | str, str, bool]:
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = "bf16-mixed"
        use_compile = True
    else:
        accelerator = "cpu"
        devices = "auto"
        precision = "32-true"
        use_compile = False

    return (accelerator, devices, precision, use_compile)
