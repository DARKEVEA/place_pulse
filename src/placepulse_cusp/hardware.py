from __future__ import annotations

import platform
import time
from typing import Any

import torch

from placepulse_cusp.models.base import select_device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _optional_mps_metric(name: str) -> int | None:
    function = getattr(torch.mps, name, None)
    return int(function()) if callable(function) else None


def device_report(requested: str = "auto") -> dict[str, Any]:
    report: dict[str, Any] = {
        "requested": requested,
        "platform": platform.platform(),
        "python_torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "mps_built": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }
    device = select_device(requested)
    report["selected"] = str(device)
    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        report["cuda"] = {
            "device_index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": properties.total_memory,
            "free_memory_bytes": free,
            "allocator_backend": torch.cuda.get_allocator_backend(),
        }
    elif device.type == "mps":
        report["mps"] = {
            "recommended_working_set_bytes": _optional_mps_metric(
                "recommended_max_memory"
            ),
            "current_allocated_bytes": _optional_mps_metric(
                "current_allocated_memory"
            ),
            "driver_allocated_bytes": _optional_mps_metric(
                "driver_allocated_memory"
            ),
        }
    return report


def gpu_smoke(requested: str = "auto", size: int = 2048, iterations: int = 5) -> dict[str, Any]:
    if size < 8:
        raise ValueError("--size must be at least 8.")
    if iterations < 1:
        raise ValueError("--iterations must be positive.")
    device = select_device(requested)
    if device.type == "cpu":
        raise RuntimeError("GPU smoke requires --device mps or --device cuda.")
    torch.manual_seed(1103)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    left = torch.randn(size, size, device=device, requires_grad=True)
    right = torch.randn(size, size, device=device)
    index = torch.arange(size, device=device) % max(size // 8, 1)
    _synchronize(device)
    started = time.perf_counter()
    loss_value = 0.0
    for _ in range(iterations):
        product = left @ right
        reduced = torch.zeros(max(size // 8, 1), size, device=device)
        reduced.index_add_(0, index, product)
        loss = reduced.square().mean()
        loss.backward()
        loss_value = float(loss.detach().cpu())
        left.grad = None
    _synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "device": str(device),
        "matrix_size": size,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "iterations_per_second": iterations / elapsed,
        "loss": loss_value,
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    elif device.type == "mps":
        result["driver_allocated_bytes"] = _optional_mps_metric(
            "driver_allocated_memory"
        )
    return result
