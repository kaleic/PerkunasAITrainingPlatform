from __future__ import annotations

import logging
from dataclasses import dataclass

import torch


logger = logging.getLogger("perkunas.device")


@dataclass(frozen=True, slots=True)
class DeviceDiagnostics:
    cuda_available: bool
    torch_cuda_version: str | None
    cuda_device_count: int
    cuda_device_name_0: str | None
    selected_device: str

    def as_dict(self) -> dict[str, object]:
        return {
            "torch.cuda.is_available": self.cuda_available,
            "torch.version.cuda": self.torch_cuda_version,
            "torch.cuda.device_count": self.cuda_device_count,
            "torch.cuda.get_device_name(0)": self.cuda_device_name_0,
            "selected_device": self.selected_device,
        }


def print_cuda_diagnostics(selected_device: torch.device | None = None) -> DeviceDiagnostics:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
    diagnostics = DeviceDiagnostics(
        cuda_available=cuda_available,
        torch_cuda_version=torch.version.cuda,
        cuda_device_count=device_count,
        cuda_device_name_0=name,
        selected_device=str(selected_device) if selected_device is not None else "unselected",
    )
    for key, value in diagnostics.as_dict().items():
        print(f"{key}: {value}", flush=True)
        logger.info("%s: %s", key, value)
    return diagnostics


def select_device(require_gpu: bool, local_rank: int = 0) -> torch.device:
    diagnostics = print_cuda_diagnostics()
    if diagnostics.cuda_available:
        torch.backends.cudnn.benchmark = True
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        if require_gpu:
            raise RuntimeError(
                "Perkunas training requires GPU but CUDA is unavailable. "
                f"torch.version.cuda={torch.version.cuda!r}, "
                f"torch.cuda.device_count()={torch.cuda.device_count()}. "
                "Install a CUDA-enabled PyTorch build and NVIDIA driver, or set require_gpu=false "
                "only for intentional CPU smoke tests."
            )
        device = torch.device("cpu")
    if diagnostics.cuda_available and device.type != "cuda":
        raise RuntimeError("CUDA is available but Perkunas selected a non-CUDA device")
    print_cuda_diagnostics(device)
    return device


def assert_model_device(model: torch.nn.Module, *, require_gpu: bool) -> torch.device:
    module = model.module if hasattr(model, "module") else model
    try:
        parameter_device = next(module.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("model has no parameters") from exc
    if require_gpu and parameter_device.type != "cuda":
        raise AssertionError(
            f"Perkunas model is on {parameter_device}, but require_gpu=true requires CUDA"
        )
    if torch.cuda.is_available() and parameter_device.type != "cuda":
        raise AssertionError(
            f"CUDA is available but Perkunas model is on {parameter_device}; refusing CPU training"
        )
    return parameter_device


def verify_batch_devices(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    expected_device: torch.device,
    require_gpu: bool,
) -> None:
    mismatches: list[str] = []
    if input_ids.device != expected_device:
        mismatches.append(f"input_ids={input_ids.device}")
    if labels.device != expected_device:
        mismatches.append(f"labels={labels.device}")
    if mismatches:
        message = (
            f"Mixed-device batch detected; expected {expected_device}, "
            + ", ".join(mismatches)
        )
        logger.warning(message)
        print(f"WARNING: {message}", flush=True)
        if require_gpu:
            raise RuntimeError(message)


def log_gpu_memory(prefix: str = "GPU memory allocated") -> None:
    if torch.cuda.is_available():
        mb = torch.cuda.memory_allocated() / 1024**2
        print(f"{prefix}: {mb:.2f} MB", flush=True)
        logger.info("%s: %.2f MB", prefix, mb)
    else:
        print(f"{prefix}: CUDA unavailable", flush=True)
        logger.info("%s: CUDA unavailable", prefix)


def log_first_batch_verification(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    require_gpu: bool,
) -> None:
    model_device = assert_model_device(model, require_gpu=require_gpu)
    print(f"Perkunas first batch model device: {model_device}", flush=True)
    print(f"Perkunas first batch input_ids device: {input_ids.device}", flush=True)
    print(f"Perkunas first batch labels device: {labels.device}", flush=True)
    print(f"Perkunas first batch size: {batch_size}", flush=True)
    log_gpu_memory("GPU memory allocated")
