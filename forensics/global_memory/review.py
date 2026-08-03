from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class IdentityReviewError(RuntimeError):
    """Base error for supervisor identity-review operations."""


class InvalidReviewRequestError(IdentityReviewError, ValueError):
    """A review query or decision contains an invalid argument."""


class ReviewSuggestionNotFoundError(IdentityReviewError):
    """The requested identity-match suggestion does not exist."""


class InvalidReviewDecisionError(InvalidReviewRequestError):
    """The requested supervisor decision is unsupported."""


class ReviewSuggestionConflictError(IdentityReviewError):
    """A persisted decision conflicts with the requested decision."""


class ReviewSuggestionStaleError(ReviewSuggestionConflictError):
    """The suggestion can no longer be decided."""


class ReviewSuggestionIntegrityError(IdentityReviewError):
    """Suggestion, person, redirect, and audit state disagree."""


class IdentityReviewDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class IdentityReviewSummary:
    suggestion_id: str
    status: str
    source_person_id: str
    candidate_person_id: str
    similarity: float
    second_similarity: float | None
    margin: float | None
    reason: str | None
    created_at: str
    reviewed_at: str | None
    reviewed_by: str | None
    source_name: str
    candidate_name: str
    source_preview: str | None
    candidate_preview: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdentityReviewDetail:
    suggestion: IdentityReviewSummary
    source_profile: dict[str, Any]
    candidate_profile: dict[str, Any]
    source_lineage: tuple[dict[str, Any], ...]
    candidate_lineage: tuple[dict[str, Any], ...]
    source_gallery: tuple[dict[str, Any], ...]
    candidate_gallery: tuple[dict[str, Any], ...]
    source_appearances: tuple[dict[str, Any], ...]
    candidate_appearances: tuple[dict[str, Any], ...]
    source_recognition_events: tuple[dict[str, Any], ...]
    candidate_recognition_events: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdentityReviewDecisionResult:
    suggestion_id: str
    decision: IdentityReviewDecision
    status: str
    source_person_id: str
    target_person_id: str
    audit_id: int | None
    idempotent_replay: bool
    staled_suggestion_count: int
    source_embedding_count: int | None
    target_embedding_count_before: int | None
    target_embedding_count_after: int | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["decision"] = self.decision.value
        return result
