from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from typing import Mapping


_MINIMUM_ENV = "PERSON_CREATION_IDENTITY_MIN_SIMILARITY"
_MAXIMUM_ENV = "PERSON_CREATION_IDENTITY_MAX_SIMILARITY"
_MARGIN_ENV = "PERSON_CREATION_IDENTITY_MIN_MARGIN"
_OBSERVATIONS_ENV = "PERSON_CREATION_IDENTITY_MIN_FACE_OBSERVATIONS"
_LEGACY_MINIMUM_ENV = "FACE_SIMILARITY_THRESHOLD"


class IdentityPolicyConfigurationError(ValueError):
    """Identity-policy configuration is invalid."""


class IdentityPolicyInputError(ValueError):
    """Inputs to an identity-policy decision are invalid."""


class IdentityDecisionType(str, Enum):
    NEW_PERSON = "new_person"
    REVIEW_REQUIRED = "review_required"
    ATTACH_EXISTING = "attach_existing"


class IdentityDecisionReason(str, Enum):
    NO_ACTIVE_CANDIDATE = "no_active_candidate"
    BELOW_MINIMUM_SIMILARITY = "below_minimum_similarity"
    SIMILARITY_BETWEEN_THRESHOLDS = "similarity_between_thresholds"
    LOW_CONFIDENCE_CLUSTER = "low_confidence_cluster"
    INSUFFICIENT_FACE_OBSERVATIONS = "insufficient_face_observations"
    CANDIDATE_MARGIN_TOO_SMALL = "candidate_margin_too_small"
    STRONG_CLEAR_MATCH = "strong_clear_match"


def _finite_float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise IdentityPolicyConfigurationError(f"{label} must be a number") from exc
    if not math.isfinite(parsed):
        raise IdentityPolicyConfigurationError(f"{label} must be finite")
    return parsed


def _environment_float(
    environ: Mapping[str, str],
    key: str,
    default: float,
) -> float:
    return _finite_float(environ[key], key) if key in environ else default


def _environment_integer(
    environ: Mapping[str, str],
    key: str,
    default: int,
) -> int:
    if key not in environ:
        return default
    try:
        return int(environ[key])
    except (TypeError, ValueError) as exc:
        raise IdentityPolicyConfigurationError(f"{key} must be an integer") from exc


@dataclass(frozen=True)
class IdentityPolicyConfig:
    minimum_similarity: float = 0.68
    maximum_similarity: float = 0.78
    minimum_margin: float = 0.03
    minimum_face_observations: int = 3

    def __post_init__(self) -> None:
        minimum = _finite_float(self.minimum_similarity, "minimum_similarity")
        maximum = _finite_float(self.maximum_similarity, "maximum_similarity")
        margin = _finite_float(self.minimum_margin, "minimum_margin")
        observations = self.minimum_face_observations
        if isinstance(observations, bool) or not isinstance(observations, int):
            raise IdentityPolicyConfigurationError(
                "minimum_face_observations must be an integer"
            )
        if not 0.0 <= minimum < maximum <= 1.0:
            raise IdentityPolicyConfigurationError(
                "identity similarities must satisfy "
                "0.0 <= minimum_similarity < maximum_similarity <= 1.0"
            )
        if not 0.0 <= margin <= 1.0:
            raise IdentityPolicyConfigurationError(
                "minimum_margin must be between 0.0 and 1.0"
            )
        if observations < 1:
            raise IdentityPolicyConfigurationError(
                "minimum_face_observations must be at least 1"
            )
        object.__setattr__(self, "minimum_similarity", minimum)
        object.__setattr__(self, "maximum_similarity", maximum)
        object.__setattr__(self, "minimum_margin", margin)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> IdentityPolicyConfig:
        values = os.environ if environ is None else environ
        if _MINIMUM_ENV in values:
            minimum = _environment_float(values, _MINIMUM_ENV, 0.68)
        elif _LEGACY_MINIMUM_ENV in values:
            minimum = _environment_float(values, _LEGACY_MINIMUM_ENV, 0.68)
        else:
            minimum = 0.68
        return cls(
            minimum_similarity=minimum,
            maximum_similarity=_environment_float(values, _MAXIMUM_ENV, 0.78),
            minimum_margin=_environment_float(values, _MARGIN_ENV, 0.03),
            minimum_face_observations=_environment_integer(
                values,
                _OBSERVATIONS_ENV,
                3,
            ),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "minimum_similarity": self.minimum_similarity,
            "maximum_similarity": self.maximum_similarity,
            "minimum_margin": self.minimum_margin,
            "minimum_face_observations": self.minimum_face_observations,
        }


@dataclass(frozen=True)
class IdentityCandidate:
    person_id: str
    similarity: float

    def __post_init__(self) -> None:
        if not str(self.person_id):
            raise IdentityPolicyInputError("candidate person_id must not be empty")
        try:
            similarity = float(self.similarity)
        except (TypeError, ValueError) as exc:
            raise IdentityPolicyInputError(
                "candidate similarity must be a number"
            ) from exc
        if not math.isfinite(similarity):
            raise IdentityPolicyInputError("candidate similarity must be finite")
        object.__setattr__(self, "person_id", str(self.person_id))
        object.__setattr__(self, "similarity", similarity)


@dataclass(frozen=True)
class IdentityDecision:
    decision: IdentityDecisionType
    reason: IdentityDecisionReason
    top_candidate_person_id: str | None
    top_similarity: float | None
    second_candidate_person_id: str | None
    second_similarity: float | None
    margin: float | None
    minimum_similarity: float
    maximum_similarity: float
    required_margin: float
    minimum_face_observations: int
    observation_count: int
    low_confidence: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "reason": self.reason.value,
            "top_candidate_person_id": self.top_candidate_person_id,
            "top_similarity": self.top_similarity,
            "second_candidate_person_id": self.second_candidate_person_id,
            "second_similarity": self.second_similarity,
            "margin": self.margin,
            "minimum_similarity": self.minimum_similarity,
            "maximum_similarity": self.maximum_similarity,
            "required_margin": self.required_margin,
            "minimum_face_observations": self.minimum_face_observations,
            "observation_count": self.observation_count,
            "low_confidence": self.low_confidence,
        }


@dataclass(frozen=True)
class IdentityRegistrationResult:
    person_id: str
    decision: IdentityDecisionType
    reason: IdentityDecisionReason
    suggestion_id: int | None
    top_candidate_person_id: str | None
    top_similarity: float | None
    second_candidate_person_id: str | None
    second_similarity: float | None
    margin: float | None
    observation_count: int
    low_confidence: bool
    configuration: IdentityPolicyConfig

    @classmethod
    def from_decision(
        cls,
        *,
        person_id: str,
        suggestion_id: int | None,
        decision: IdentityDecision,
        configuration: IdentityPolicyConfig,
    ) -> IdentityRegistrationResult:
        return cls(
            person_id=person_id,
            decision=decision.decision,
            reason=decision.reason,
            suggestion_id=suggestion_id,
            top_candidate_person_id=decision.top_candidate_person_id,
            top_similarity=decision.top_similarity,
            second_candidate_person_id=decision.second_candidate_person_id,
            second_similarity=decision.second_similarity,
            margin=decision.margin,
            observation_count=decision.observation_count,
            low_confidence=decision.low_confidence,
            configuration=configuration,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "person_id": self.person_id,
            "decision": self.decision.value,
            "reason": self.reason.value,
            "suggestion_id": self.suggestion_id,
            "top_candidate_person_id": self.top_candidate_person_id,
            "top_similarity": self.top_similarity,
            "second_candidate_person_id": self.second_candidate_person_id,
            "second_similarity": self.second_similarity,
            "margin": self.margin,
            "observation_count": self.observation_count,
            "low_confidence": self.low_confidence,
            "configuration": self.configuration.as_dict(),
        }


@dataclass(frozen=True)
class IdentityEvidenceAppendResult:
    """Outcome of appending later evidence to one canonical identity."""

    person_id: str
    canonical_person_id: str
    appended: bool
    idempotent_replay: bool
    appended_evidence_keys: tuple[str, ...]
    embedding_count_before: int
    embedding_count_after: int
    gallery_rows_added: int

    def as_dict(self) -> dict[str, object]:
        return {
            "person_id": self.person_id,
            "canonical_person_id": self.canonical_person_id,
            "appended": self.appended,
            "idempotent_replay": self.idempotent_replay,
            "appended_evidence_keys": list(self.appended_evidence_keys),
            "embedding_count_before": self.embedding_count_before,
            "embedding_count_after": self.embedding_count_after,
            "gallery_rows_added": self.gallery_rows_added,
        }


def evaluate_identity_decision(
    *,
    top_candidate: IdentityCandidate | None,
    second_candidate: IdentityCandidate | None,
    observation_count: int,
    low_confidence: bool,
    configuration: IdentityPolicyConfig,
) -> IdentityDecision:
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise IdentityPolicyInputError("observation_count must be an integer")
    if observation_count < 0:
        raise IdentityPolicyInputError("observation_count must not be negative")
    if not isinstance(low_confidence, bool):
        raise IdentityPolicyInputError("low_confidence must be a boolean")
    if not isinstance(configuration, IdentityPolicyConfig):
        raise IdentityPolicyInputError(
            "configuration must be an IdentityPolicyConfig"
        )
    if top_candidate is None and second_candidate is not None:
        raise IdentityPolicyInputError(
            "second_candidate cannot be provided without top_candidate"
        )

    top_id = top_candidate.person_id if top_candidate is not None else None
    top_similarity = (
        top_candidate.similarity if top_candidate is not None else None
    )
    second_id = (
        second_candidate.person_id if second_candidate is not None else None
    )
    second_similarity = (
        second_candidate.similarity if second_candidate is not None else None
    )
    margin = (
        top_similarity - second_similarity
        if top_similarity is not None and second_similarity is not None
        else None
    )
    if margin is not None and not math.isfinite(margin):
        raise IdentityPolicyInputError("candidate margin must be finite")
    margin_is_sufficient = (
        margin is None
        or margin > configuration.minimum_margin
        or math.isclose(
            margin,
            configuration.minimum_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )

    if top_candidate is None:
        decision = IdentityDecisionType.NEW_PERSON
        reason = IdentityDecisionReason.NO_ACTIVE_CANDIDATE
    elif top_candidate.similarity < configuration.minimum_similarity:
        decision = IdentityDecisionType.NEW_PERSON
        reason = IdentityDecisionReason.BELOW_MINIMUM_SIMILARITY
    elif top_candidate.similarity < configuration.maximum_similarity:
        decision = IdentityDecisionType.REVIEW_REQUIRED
        reason = IdentityDecisionReason.SIMILARITY_BETWEEN_THRESHOLDS
    elif low_confidence:
        decision = IdentityDecisionType.REVIEW_REQUIRED
        reason = IdentityDecisionReason.LOW_CONFIDENCE_CLUSTER
    elif observation_count < configuration.minimum_face_observations:
        decision = IdentityDecisionType.REVIEW_REQUIRED
        reason = IdentityDecisionReason.INSUFFICIENT_FACE_OBSERVATIONS
    elif not margin_is_sufficient:
        decision = IdentityDecisionType.REVIEW_REQUIRED
        reason = IdentityDecisionReason.CANDIDATE_MARGIN_TOO_SMALL
    else:
        decision = IdentityDecisionType.ATTACH_EXISTING
        reason = IdentityDecisionReason.STRONG_CLEAR_MATCH

    return IdentityDecision(
        decision=decision,
        reason=reason,
        top_candidate_person_id=top_id,
        top_similarity=top_similarity,
        second_candidate_person_id=second_id,
        second_similarity=second_similarity,
        margin=margin,
        minimum_similarity=configuration.minimum_similarity,
        maximum_similarity=configuration.maximum_similarity,
        required_margin=configuration.minimum_margin,
        minimum_face_observations=configuration.minimum_face_observations,
        observation_count=observation_count,
        low_confidence=low_confidence,
    )
