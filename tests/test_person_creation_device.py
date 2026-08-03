from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from forensics.person_creation.models.device import resolve_device


def _mock_cuda(monkeypatch, available: bool) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: available),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_explicit_cpu_overrides_environment_cuda(monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_DEVICE", "cuda")
    _mock_cuda(monkeypatch, available=True)

    assert resolve_device("cpu") == "cpu"


def test_explicit_cuda_overrides_environment_cpu(monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_DEVICE", "cpu")
    _mock_cuda(monkeypatch, available=True)

    assert resolve_device("cuda") == "cuda:0"


def test_auto_uses_person_creation_device_cpu(monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_DEVICE", "cpu")
    _mock_cuda(monkeypatch, available=True)

    assert resolve_device("auto") == "cpu"


def test_auto_uses_person_creation_device_cuda(monkeypatch):
    monkeypatch.setenv("PERSON_CREATION_DEVICE", "cuda")
    _mock_cuda(monkeypatch, available=True)

    assert resolve_device("auto") == "cuda:0"


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (True, "cuda:0"),
        (False, "cpu"),
    ],
)
def test_missing_environment_with_auto_selects_based_on_cuda(monkeypatch, available, expected):
    monkeypatch.delenv("PERSON_CREATION_DEVICE", raising=False)
    _mock_cuda(monkeypatch, available=available)

    assert resolve_device("auto") == expected


def test_invalid_device_raises_clear_value_error(monkeypatch):
    monkeypatch.delenv("PERSON_CREATION_DEVICE", raising=False)

    with pytest.raises(ValueError, match="expected auto, cpu, cuda, or cuda:<index>"):
        resolve_device("gpu")


def test_explicit_cuda_while_unavailable_raises_clear_runtime_error(monkeypatch):
    monkeypatch.delenv("PERSON_CREATION_DEVICE", raising=False)
    _mock_cuda(monkeypatch, available=False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")
