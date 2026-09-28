from __future__ import annotations

from .model import TextToLatentRFDiT
from .rf import RFVelocityFn


def create_rf_dit_backend(
    model: TextToLatentRFDiT,
    *,
    device: str,
) -> RFVelocityFn:
    del model, device
    raise RuntimeError(
        "model_device='npu' is wired but the OpenVINO RF-DiT backend is not implemented yet."
    )
