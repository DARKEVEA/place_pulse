import pytest
import torch

from placepulse_cusp.hardware import device_report, gpu_smoke
from placepulse_cusp.models.base import select_device


def test_cpu_device_report_is_available():
    report = device_report("cpu")
    assert report["selected"] == "cpu"
    assert "cuda_available" in report
    assert "mps_available" in report


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        select_device("quantum")


def test_gpu_smoke_rejects_cpu():
    with pytest.raises(RuntimeError):
        gpu_smoke("cpu", size=8, iterations=1)


def test_gpu_smoke_validates_workload_before_device():
    with pytest.raises(ValueError, match="size"):
        gpu_smoke("cpu", size=4, iterations=1)
    with pytest.raises(ValueError, match="iterations"):
        gpu_smoke("cpu", size=8, iterations=0)


def test_explicit_unavailable_cuda_has_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda")
