from __future__ import annotations

from typing import Any

from kvserve.models.schemas import ModelSpec, QuantizationMode


def build_transformers_quantization_config(model: ModelSpec) -> Any | None:
    mode = model.quantization_mode
    if mode in {QuantizationMode.BF16, QuantizationMode.FP16, QuantizationMode.FP8, QuantizationMode.AUTO}:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            f"{mode.value} online quantization for Transformers requires transformers bitsandbytes support"
        ) from exc

    if mode == QuantizationMode.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == QuantizationMode.INT4:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("INT4 online quantization requires torch") from exc
        dtype_name = model.backend_config.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = getattr(torch, dtype_name)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model.backend_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    return None
