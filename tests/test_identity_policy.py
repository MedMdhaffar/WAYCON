from __future__ import annotations

import math

import pytest

from forensics.global_memory.identity_policy import (
    IdentityCandidate,
    IdentityDecisionReason,
    IdentityDecisionType,
    IdentityPolicyConfig,
    IdentityPolicyConfigurationError,
    IdentityPolicyInputError,
    evaluate_identity_decision,
)


DEFAULT = IdentityPolicyConfig()


def _candidate(person_id: str, similarity: float) -> IdentityCandidate:
    return IdentityCandidate(person_id=person_id, similarity=similarity)


def _evaluate(
    top: IdentityCandidate | None,
    second: IdentityCandidate | None = None,
    *,
    observations: int = 3,
    low_confidence: bool = False,
    configuration: IdentityPolicyConfig = DEFAULT,
):
    return evaluate_identity_decision(
        top_candidate=top,
        second_candidate=second,
        observation_count=observations,
        low_confidence=low_confidence,
        configuration=configuration,
    )


def test_no_candidate_returns_new_person_with_null_candidate_values():
    result = _evaluate(None)

    assert result.decision is IdentityDecisionType.NEW_PERSON
    assert result.reason is IdentityDecisionReason.NO_ACTIVE_CANDIDATE
    assert result.top_candidate_person_id is None
    assert result.top_similarity is None
    assert result.second_candidate_person_id is None
    assert result.second_similarity is None
    assert result.margin is None


@pytest.mark.parametrize(
    ("similarity", "decision", "reason"),
    [
        (
            0.679999,
            IdentityDecisionType.NEW_PERSON,
            IdentityDecisionReason.BELOW_MINIMUM_SIMILARITY,
        ),
        (
            0.68,
            IdentityDecisionType.REVIEW_REQUIRED,
            IdentityDecisionReason.SIMILARITY_BETWEEN_THRESHOLDS,
        ),
        (
            0.73,
            IdentityDecisionType.REVIEW_REQUIRED,
            IdentityDecisionReason.SIMILARITY_BETWEEN_THRESHOLDS,
        ),
        (
            0.779999,
            IdentityDecisionType.REVIEW_REQUIRED,
            IdentityDecisionReason.SIMILARITY_BETWEEN_THRESHOLDS,
        ),
        (
            0.78,
            IdentityDecisionType.ATTACH_EXISTING,
            IdentityDecisionReason.STRONG_CLEAR_MATCH,
        ),
        (
            0.91,
            IdentityDecisionType.ATTACH_EXISTING,
            IdentityDecisionReason.STRONG_CLEAR_MATCH,
        ),
    ],
)
def test_similarity_threshold_boundaries(similarity, decision, reason):
    result = _evaluate(_candidate("person_001", similarity))

    assert result.decision is decision
    assert result.reason is reason


def test_low_confidence_blocks_high_similarity_attachment():
    result = _evaluate(_candidate("person_001", 0.95), low_confidence=True)

    assert result.decision is IdentityDecisionType.REVIEW_REQUIRED
    assert result.reason is IdentityDecisionReason.LOW_CONFIDENCE_CLUSTER


def test_insufficient_observations_block_high_similarity_attachment():
    result = _evaluate(_candidate("person_001", 0.95), observations=2)

    assert result.decision is IdentityDecisionType.REVIEW_REQUIRED
    assert result.reason is IdentityDecisionReason.INSUFFICIENT_FACE_OBSERVATIONS


def test_observation_count_exactly_three_passes():
    result = _evaluate(_candidate("person_001", 0.95), observations=3)

    assert result.decision is IdentityDecisionType.ATTACH_EXISTING
    assert result.reason is IdentityDecisionReason.STRONG_CLEAR_MATCH


@pytest.mark.parametrize(
    ("second_similarity", "decision", "reason"),
    [
        (
            0.870001,
            IdentityDecisionType.REVIEW_REQUIRED,
            IdentityDecisionReason.CANDIDATE_MARGIN_TOO_SMALL,
        ),
        (
            0.87,
            IdentityDecisionType.ATTACH_EXISTING,
            IdentityDecisionReason.STRONG_CLEAR_MATCH,
        ),
        (
            0.86,
            IdentityDecisionType.ATTACH_EXISTING,
            IdentityDecisionReason.STRONG_CLEAR_MATCH,
        ),
    ],
)
def test_candidate_margin_boundaries(second_similarity, decision, reason):
    result = _evaluate(
        _candidate("person_001", 0.90),
        _candidate("person_002", second_similarity),
    )

    assert result.decision is decision
    assert result.reason is reason


def test_margin_exactly_point_zero_three_passes():
    configuration = IdentityPolicyConfig(
        minimum_similarity=0.01,
        maximum_similarity=0.03,
        minimum_margin=0.03,
        minimum_face_observations=3,
    )

    result = _evaluate(
        _candidate("person_001", 0.03),
        _candidate("person_002", 0.0),
        configuration=configuration,
    )

    assert result.margin == 0.03
    assert result.decision is IdentityDecisionType.ATTACH_EXISTING
    assert result.reason is IdentityDecisionReason.STRONG_CLEAR_MATCH


@pytest.mark.parametrize(
    ("top_similarity", "second_similarity"),
    [
        (0.81, 0.78),
        (0.82, 0.79),
        (0.80, 0.77),
        (0.83, 0.80),
    ],
)
def test_decimal_exact_margin_representations_pass(
    top_similarity,
    second_similarity,
):
    result = _evaluate(
        _candidate("person_001", top_similarity),
        _candidate("person_002", second_similarity),
    )

    assert result.margin == top_similarity - second_similarity
    assert result.decision is IdentityDecisionType.ATTACH_EXISTING
    assert result.reason is IdentityDecisionReason.STRONG_CLEAR_MATCH


def test_exact_margin_tolerance_preserves_raw_subtraction_result():
    result = _evaluate(
        _candidate("person_001", 0.82),
        _candidate("person_002", 0.79),
    )

    assert result.margin == 0.82 - 0.79
    assert result.margin == 0.029999999999999916
    assert result.margin != 0.03
    assert result.decision is IdentityDecisionType.ATTACH_EXISTING


@pytest.mark.parametrize(
    ("second_similarity", "decision", "reason"),
    [
        (
            0.7900001,
            IdentityDecisionType.REVIEW_REQUIRED,
            IdentityDecisionReason.CANDIDATE_MARGIN_TOO_SMALL,
        ),
        (
            0.7899999,
            IdentityDecisionType.ATTACH_EXISTING,
            IdentityDecisionReason.STRONG_CLEAR_MATCH,
        ),
    ],
)
def test_margin_tolerance_does_not_change_genuine_near_boundary_values(
    second_similarity,
    decision,
    reason,
):
    result = _evaluate(
        _candidate("person_001", 0.82),
        _candidate("person_002", second_similarity),
    )

    assert result.margin == 0.82 - second_similarity
    assert result.decision is decision
    assert result.reason is reason


@pytest.mark.parametrize(
    ("observations", "low_confidence", "reason"),
    [
        (3, True, IdentityDecisionReason.LOW_CONFIDENCE_CLUSTER),
        (2, False, IdentityDecisionReason.INSUFFICIENT_FACE_OBSERVATIONS),
    ],
)
def test_exact_margin_does_not_bypass_earlier_safety_gates(
    observations,
    low_confidence,
    reason,
):
    result = _evaluate(
        _candidate("person_001", 0.82),
        _candidate("person_002", 0.79),
        observations=observations,
        low_confidence=low_confidence,
    )

    assert result.decision is IdentityDecisionType.REVIEW_REQUIRED
    assert result.reason is reason


def test_missing_second_candidate_passes_margin_gate_with_null_values():
    result = _evaluate(_candidate("person_001", 0.90))

    assert result.decision is IdentityDecisionType.ATTACH_EXISTING
    assert result.second_candidate_person_id is None
    assert result.second_similarity is None
    assert result.margin is None


def test_high_similarity_reason_precedence_is_deterministic():
    result = _evaluate(
        _candidate("person_001", 0.90),
        _candidate("person_002", 0.89),
        observations=1,
        low_confidence=True,
    )

    assert result.decision is IdentityDecisionType.REVIEW_REQUIRED
    assert result.reason is IdentityDecisionReason.LOW_CONFIDENCE_CLUSTER


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_similarity": 0.78, "maximum_similarity": 0.78},
        {"minimum_similarity": 0.79, "maximum_similarity": 0.78},
        {"minimum_margin": -0.01},
        {"minimum_margin": 1.01},
        {"minimum_face_observations": 0},
        {"minimum_face_observations": -1},
        {"minimum_face_observations": True},
        {"minimum_similarity": float("nan")},
        {"maximum_similarity": float("inf")},
        {"minimum_margin": float("-inf")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(IdentityPolicyConfigurationError):
        IdentityPolicyConfig(**kwargs)


def test_environment_defaults_and_legacy_minimum_precedence():
    assert IdentityPolicyConfig.from_environment({}) == IdentityPolicyConfig(
        minimum_similarity=0.68,
        maximum_similarity=0.78,
        minimum_margin=0.03,
        minimum_face_observations=3,
    )
    legacy = IdentityPolicyConfig.from_environment(
        {"FACE_SIMILARITY_THRESHOLD": "0.70"}
    )
    explicit = IdentityPolicyConfig.from_environment(
        {
            "FACE_SIMILARITY_THRESHOLD": "0.70",
            "PERSON_CREATION_IDENTITY_MIN_SIMILARITY": "0.69",
        }
    )

    assert legacy.minimum_similarity == 0.70
    assert explicit.minimum_similarity == 0.69


def test_all_environment_values_are_loaded_and_validated():
    result = IdentityPolicyConfig.from_environment(
        {
            "PERSON_CREATION_IDENTITY_MIN_SIMILARITY": "0.67",
            "PERSON_CREATION_IDENTITY_MAX_SIMILARITY": "0.81",
            "PERSON_CREATION_IDENTITY_MIN_MARGIN": "0.04",
            "PERSON_CREATION_IDENTITY_MIN_FACE_OBSERVATIONS": "4",
        }
    )

    assert result == IdentityPolicyConfig(0.67, 0.81, 0.04, 4)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_environment_values_are_rejected(value):
    with pytest.raises(IdentityPolicyConfigurationError, match="finite"):
        IdentityPolicyConfig.from_environment(
            {"PERSON_CREATION_IDENTITY_MIN_SIMILARITY": value}
        )


def test_complete_structured_output():
    result = _evaluate(
        _candidate("person_001", 0.90),
        _candidate("person_002", 0.85),
        observations=4,
    )

    assert result.as_dict() == {
        "decision": "attach_existing",
        "reason": "strong_clear_match",
        "top_candidate_person_id": "person_001",
        "top_similarity": 0.90,
        "second_candidate_person_id": "person_002",
        "second_similarity": 0.85,
        "margin": 0.050000000000000044,
        "minimum_similarity": 0.68,
        "maximum_similarity": 0.78,
        "required_margin": 0.03,
        "minimum_face_observations": 3,
        "observation_count": 4,
        "low_confidence": False,
    }


def test_candidate_inputs_are_not_mutated_and_tie_order_is_preserved():
    top = _candidate("person_001", 0.90)
    second = _candidate("person_002", 0.90)
    before = (top, second)

    result = _evaluate(top, second)

    assert (top, second) == before
    assert result.top_candidate_person_id == "person_001"
    assert result.second_candidate_person_id == "person_002"
    assert result.reason is IdentityDecisionReason.CANDIDATE_MARGIN_TOO_SMALL


@pytest.mark.parametrize("observations", [True, -1, 1.5])
def test_invalid_observation_count_is_rejected(observations):
    with pytest.raises(IdentityPolicyInputError):
        _evaluate(_candidate("person_001", 0.90), observations=observations)


@pytest.mark.parametrize("similarity", [math.nan, math.inf, -math.inf])
def test_candidate_similarity_must_be_finite(similarity):
    with pytest.raises(IdentityPolicyInputError, match="finite"):
        _candidate("person_001", similarity)


def test_candidate_margin_must_remain_finite():
    with pytest.raises(IdentityPolicyInputError, match="margin must be finite"):
        _evaluate(
            _candidate("person_001", 1e308),
            _candidate("person_002", -1e308),
        )
