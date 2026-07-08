from __future__ import annotations

from typing import Any


def error_response(message: str, status_code: int = 400):
    from flask import jsonify

    return jsonify({"ok": False, "error": str(message)}), int(status_code)


def require_json() -> dict[str, Any]:
    from flask import request

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def require_path_field(body: dict[str, Any], field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value


def parse_top_k(body: dict[str, Any], default: int = 5, max_top_k: int = 100) -> int:
    raw = body.get("top_k", default)
    try:
        top_k = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    return min(top_k, max_top_k)


def require_embedding(value: Any, field: str = "embedding", expected_dim: int = 512) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if len(value) != expected_dim:
        raise ValueError(f"{field} must contain {expected_dim} floats")
    embedding: list[float] = []
    for idx, item in enumerate(value):
        try:
            embedding.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{idx}] must be numeric") from exc
    return embedding
