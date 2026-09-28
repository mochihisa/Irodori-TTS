from __future__ import annotations

from typing import Any

import torch

from .model import TextToLatentRFDiT
from .rf import RFVelocityFn


class _RFDiTExportWrapper(torch.nn.Module):
    def __init__(self, model: TextToLatentRFDiT) -> None:
        super().__init__()
        self.model = model
        self.use_speaker = model.cfg.use_speaker_condition_resolved
        self.use_caption = model.cfg.use_caption_condition

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor,
        speaker_mask: torch.Tensor,
        caption_state: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state if self.use_speaker else None,
            speaker_mask=speaker_mask if self.use_speaker else None,
            caption_state=caption_state if self.use_caption else None,
            caption_mask=caption_mask if self.use_caption else None,
            context_kv_cache=None,
        )


class OpenVINORFDiTBackend:
    def __init__(self, model: TextToLatentRFDiT, *, device: str) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO is required for model_device='npu'. Install the openvino package."
            ) from exc

        self.model = model.eval()
        self.device = str(device).strip().upper()
        self.core = ov.Core()
        if self.device not in self.core.available_devices:
            available = ", ".join(self.core.available_devices) or "none"
            raise RuntimeError(
                f"OpenVINO device {self.device!r} is unavailable. Available devices: {available}."
            )
        self._compiled_models: dict[
            tuple[tuple[tuple[int, ...], torch.dtype], ...], Any
        ] = {}

    def _prepare_inputs(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        batch = x_t.shape[0]
        if self.model.cfg.use_speaker_condition_resolved:
            if speaker_state is None or speaker_mask is None:
                raise ValueError("Speaker state and mask are required by this RF-DiT model.")
        else:
            speaker_state = torch.empty((batch, 1, 1), dtype=x_t.dtype, device=x_t.device)
            speaker_mask = torch.zeros((batch, 1), dtype=torch.bool, device=x_t.device)
        if self.model.cfg.use_caption_condition:
            if caption_state is None or caption_mask is None:
                raise ValueError("Caption state and mask are required by this RF-DiT model.")
        else:
            caption_state = torch.empty((batch, 1, 1), dtype=x_t.dtype, device=x_t.device)
            caption_mask = torch.zeros((batch, 1), dtype=torch.bool, device=x_t.device)
        return (
            x_t,
            t,
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        )

    def _compile(self, inputs: tuple[torch.Tensor, ...]):
        import openvino as ov

        wrapper = _RFDiTExportWrapper(self.model).eval()
        with torch.inference_mode():
            exported = torch.export.export(wrapper, inputs)
        ov_model = ov.convert_model(exported)
        ov_model.reshape(
            {
                port.any_name: list(tensor.shape)
                for port, tensor in zip(ov_model.inputs, inputs, strict=True)
            }
        )
        return self.core.compile_model(ov_model, self.device)

    def __call__(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_state: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        caption_state: torch.Tensor | None = None,
        caption_mask: torch.Tensor | None = None,
        context_kv_cache: list[tuple[torch.Tensor, ...]] | None = None,
    ) -> torch.Tensor:
        del context_kv_cache
        inputs = self._prepare_inputs(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=caption_state,
            caption_mask=caption_mask,
        )
        signature = tuple((tuple(tensor.shape), tensor.dtype) for tensor in inputs)
        compiled_model = self._compiled_models.get(signature)
        if compiled_model is None:
            compiled_model = self._compile(inputs)
            self._compiled_models[signature] = compiled_model
        output = compiled_model(inputs)[0]
        return torch.from_numpy(output.copy()).to(device=x_t.device, dtype=x_t.dtype)


def create_rf_dit_backend(
    model: TextToLatentRFDiT,
    *,
    device: str,
) -> RFVelocityFn:
    return OpenVINORFDiTBackend(model, device=device)
