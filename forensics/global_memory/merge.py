from __future__ import annotations

from dataclasses import dataclass


class PersonMergeError(RuntimeError):
    """Base error for canonical lineage and person merging."""


class InvalidMergeRequestError(PersonMergeError, ValueError):
    """A merge or lineage request contains an invalid argument."""


class PersonNotFoundError(PersonMergeError):
    """A requested person does not exist."""


class SelfMergeError(InvalidMergeRequestError):
    """A person cannot be merged into itself."""


class InactiveSourceError(PersonMergeError):
    """A source is inactive without a valid direct merge redirect."""


class InactiveTargetError(PersonMergeError):
    """A merge target is not an active canonical person."""


class AlreadyMergedConflictError(PersonMergeError):
    """A source already redirects to a different canonical target."""


class RedirectChainError(PersonMergeError):
    """Identity redirects contain or would create a chain or cycle."""


class InvalidMergeEmbeddingError(PersonMergeError, ValueError):
    """Stored biometric evidence cannot be safely consolidated."""


class InvalidMergeMetadataError(PersonMergeError, ValueError):
    """Stored identity metadata cannot be safely consolidated."""


class MergeAuditIntegrityError(PersonMergeError):
    """Persisted redirect and immutable merge-audit state disagree."""


@dataclass(frozen=True)
class IdentityLineageMember:
    person_id: str
    is_active: bool
    merged_into_person_id: str | None


@dataclass(frozen=True)
class IdentityLineage:
    canonical_person_id: str
    members: tuple[IdentityLineageMember, ...]

    @property
    def member_person_ids(self) -> tuple[str, ...]:
        return tuple(member.person_id for member in self.members)


@dataclass(frozen=True)
class PersonMergeResult:
    source_person_id: str
    target_person_id: str
    audit_id: int
    idempotent_replay: bool
    source_embedding_count: int
    target_embedding_count_before: int
    target_embedding_count_after: int
    staled_suggestion_count: int
    lineage_member_count_after: int
