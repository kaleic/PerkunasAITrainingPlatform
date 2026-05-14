from __future__ import annotations

import logging
from typing import Any

from kvserve.config import Settings


logger = logging.getLogger("kvserve.gpu")


def configure_cuda_runtime(settings: Settings) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        if settings.require_cuda:
            raise RuntimeError("KV_REQUIRE_CUDA=true but torch is not installed") from exc
        logger.info("torch unavailable; CUDA runtime not configured")
        return {"torch_available": False, "cuda_available": False}

    diagnostics: dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "torch.cuda.is_available": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.cuda.device_count": torch.cuda.device_count(),
        "torch.cuda.get_device_name(0)": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    for key, value in diagnostics.items():
        logger.info("%s=%s", key, value)
        print(f"{key}: {value}", flush=True)

    if settings.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "KV_REQUIRE_CUDA=true but CUDA is unavailable. "
            f"torch.version.cuda={torch.version.cuda!r}, "
            f"torch.cuda.device_count()={torch.cuda.device_count()}."
        )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
        diagnostics["selected_device"] = "cuda:0"
        if settings.warm_cuda:
            warm_tensor = torch.empty((256, 256), device="cuda")
            warm_tensor.fill_(1.0)
            diagnostics["warm_cuda_tensor"] = warm_tensor
            diagnostics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            print(
                f"GPU memory allocated after server warmup: "
                f"{diagnostics['gpu_memory_allocated_mb']:.2f} MB",
                flush=True,
            )
            logger.info(
                "GPU memory allocated after server warmup: %.2f MB",
                diagnostics["gpu_memory_allocated_mb"],
            )
    else:
        diagnostics["selected_device"] = "cpu"
    return diagnostics
