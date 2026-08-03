from __future__ import annotations

from types import SimpleNamespace

import pytest

from forensics.person_creation.models.device import (
    get_device_status,
    model_parameter_device,
    set_device_status,
    validate_model_devices,
)


def _devices(**overrides: str) -> dict[str, str]:
    devices = {
        "person_detector": "cpu",
        "clothing_describer": "cpu",
        "pose_estimator": "cpu",
        "reid_extractor": "cpu",
        "body_reid": "not_loaded",
    }
    devices.update(overrides)
    return devices


def test_resolved_cpu_with_all_loaded_models_on_cpu_passes():
    validate_model_devices(
        requested="auto",
        resolved="cpu",
        models=_devices(),
    )


def test_resolved_cpu_with_person_detector_on_cuda_raises():
    with pytest.raises(RuntimeError, match="person_detector.*actual='cuda:0'"):
        validate_model_devices(
            requested="auto",
            resolved="cpu",
            models=_devices(person_detector="cuda:0"),
        )


def test_resolved_cpu_with_clothing_describer_on_cuda_raises():
    with pytest.raises(RuntimeError, match="clothing_describer.*actual='cuda:0'"):
        validate_model_devices(
            requested="cpu",
            resolved="cpu",
            models=_devices(clothing_describer="cuda:0"),
        )


def test_disabled_or_unavailable_optional_models_do_not_fail():
    validate_model_devices(
        requested="auto",
        resolved="cpu",
        models=_devices(
            pose_estimator="disabled",
            reid_extractor="unavailable",
            body_reid="not_loaded",
        ),
    )


@pytest.mark.parametrize("actual", ["cuda", "cuda:0"])
def test_cuda_resolution_accepts_cuda_device_spellings(actual):
    validate_model_devices(
        requested="cuda",
        resolved="cuda:0",
        models=_devices(person_detector=actual),
    )


def test_model_parameter_device_reads_actual_parameter_device():
    parameter = SimpleNamespace(device="cuda:0")
    model = SimpleNamespace(parameters=lambda: iter([parameter]))

    assert model_parameter_device(model, fallback="cpu") == "cuda:0"


def test_model_parameter_device_surfaces_cuda_from_mixed_hf_device_map():
    model = SimpleNamespace(hf_device_map={"encoder": "cpu", "decoder": "cuda:0"})

    assert model_parameter_device(model, fallback="cpu") == "cuda:0"


def test_health_preserves_existing_fields_and_reports_devices():
    from forensics.person_creation.service import app

    set_device_status(
        requested="auto",
        environment="cpu",
        resolved="cpu",
        models=_devices(),
        models_loaded=True,
    )

    response = app.test_client().get("/api/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["service"] == "waycon-person-creation"
    assert payload["models_loaded"] is True
    assert payload["device"] == {
        "requested": "auto",
        "environment": "cpu",
        "resolved": "cpu",
        "models": _devices(),
    }
    assert get_device_status()["device"]["models"]["person_detector"] == "cpu"
