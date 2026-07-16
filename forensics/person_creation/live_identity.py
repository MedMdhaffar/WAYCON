"""Deterministic session-only identity stabilization by crop membership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class CurrentMembership:
    cluster_label: int
    crop_paths: frozenset[str]

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(sorted(self.crop_paths))


@dataclass(frozen=True)
class StableIdentity:
    session_person_id: str
    cluster_label: int
    crop_paths: frozenset[str]
    is_new: bool


@dataclass(frozen=True)
class IdentityTransition:
    event_type: str
    session_person_id: str
    previous_live_id: str | None = None
    retained_live_id: str | None = None
    child_live_ids: tuple[str, ...] = ()
    absorbed_live_ids: tuple[str, ...] = ()
    overlap_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class IdentityAssignmentResult:
    identities: tuple[StableIdentity, ...]
    active_memberships: tuple[tuple[str, frozenset[str]], ...]
    transitions: tuple[IdentityTransition, ...]
    next_live_number: int


def _live_number(live_id: str) -> int:
    try:
        return int(str(live_id).rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 2**31 - 1


def _maximum_overlap(
    currents: Sequence[CurrentMembership],
    previous_ids: Sequence[str],
    previous: Mapping[str, frozenset[str]],
) -> int:
    if not currents or not previous_ids:
        return 0
    weights = np.zeros((len(currents), len(previous_ids) + len(currents)), dtype=np.int64)
    for row, current in enumerate(currents):
        for column, live_id in enumerate(previous_ids):
            weights[row, column] = len(current.crop_paths & previous[live_id])
    rows, columns = linear_sum_assignment(-weights)
    return sum(
        int(weights[row, column])
        for row, column in zip(rows, columns)
        if column < len(previous_ids) and weights[row, column] > 0
    )


def maximum_overlap_assignment(
    current_memberships: Sequence[CurrentMembership],
    previous_memberships: Mapping[str, frozenset[str]],
) -> tuple[tuple[CurrentMembership, str | None], ...]:
    """Globally maximize retained overlap, then resolve ties lexicographically."""
    currents = sorted(current_memberships, key=lambda item: item.signature)
    previous = {
        str(live_id): frozenset(paths)
        for live_id, paths in previous_memberships.items()
    }
    available = sorted(previous, key=lambda live_id: (_live_number(live_id), live_id))
    target = _maximum_overlap(currents, available, previous)
    retained = 0
    assignments: list[tuple[CurrentMembership, str | None]] = []

    for index, current in enumerate(currents):
        remaining_currents = currents[index + 1:]
        candidates = [
            live_id
            for live_id in available
            if current.crop_paths & previous[live_id]
        ]
        candidates.append(None)
        selected: str | None = None
        for candidate in candidates:
            overlap = (
                len(current.crop_paths & previous[candidate])
                if candidate is not None
                else 0
            )
            remaining_ids = [
                live_id for live_id in available if live_id != candidate
            ]
            possible = retained + overlap + _maximum_overlap(
                remaining_currents,
                remaining_ids,
                previous,
            )
            if possible == target:
                selected = candidate
                retained += overlap
                if candidate is not None:
                    available.remove(candidate)
                break
        assignments.append((current, selected))

    return tuple(assignments)


def assign_live_identities(
    *,
    current_memberships: Sequence[CurrentMembership],
    previous_memberships: Mapping[str, frozenset[str]],
    next_live_number: int,
) -> IdentityAssignmentResult:
    assignments = maximum_overlap_assignment(
        current_memberships,
        previous_memberships,
    )
    next_number = max(1, int(next_live_number))
    identities: list[StableIdentity] = []
    assigned_previous: set[str] = set()

    for current, previous_id in assignments:
        is_new = previous_id is None
        live_id = previous_id
        if live_id is None:
            live_id = f"live_{next_number:04d}"
            next_number += 1
        else:
            assigned_previous.add(live_id)
        identities.append(StableIdentity(
            session_person_id=live_id,
            cluster_label=current.cluster_label,
            crop_paths=current.crop_paths,
            is_new=is_new,
        ))

    identities_by_signature = {
        tuple(sorted(identity.crop_paths)): identity for identity in identities
    }
    transitions: list[IdentityTransition] = [
        IdentityTransition(
            event_type="identity_created",
            session_person_id=identity.session_person_id,
        )
        for identity in identities
        if identity.is_new
    ]

    previous = {
        str(live_id): frozenset(paths)
        for live_id, paths in previous_memberships.items()
    }
    for previous_id in sorted(previous, key=lambda item: (_live_number(item), item)):
        overlaps: list[tuple[StableIdentity, int]] = []
        for current in current_memberships:
            count = len(previous[previous_id] & current.crop_paths)
            if count:
                identity = identities_by_signature[current.signature]
                overlaps.append((identity, count))
        retained = next(
            (
                identity
                for identity, _count in overlaps
                if identity.session_person_id == previous_id
            ),
            None,
        )
        children = tuple(sorted(
            (
                identity.session_person_id
                for identity, _count in overlaps
                if identity.is_new
            ),
            key=lambda item: (_live_number(item), item),
        ))
        if len(overlaps) > 1 and retained is not None and children:
            relevant = {previous_id, *children}
            transitions.append(IdentityTransition(
                event_type="identity_split",
                session_person_id=previous_id,
                previous_live_id=previous_id,
                retained_live_id=previous_id,
                child_live_ids=children,
                overlap_counts=tuple(sorted(
                    (
                        (identity.session_person_id, count)
                        for identity, count in overlaps
                        if identity.session_person_id in relevant
                    ),
                    key=lambda item: (_live_number(item[0]), item[0]),
                )),
            ))

    for identity in sorted(
        identities,
        key=lambda item: (_live_number(item.session_person_id), item.session_person_id),
    ):
        if identity.is_new:
            continue
        overlaps = []
        for previous_id, paths in previous.items():
            count = len(identity.crop_paths & paths)
            if count:
                overlaps.append((previous_id, count))
        if len(overlaps) > 1:
            absorbed = tuple(sorted(
                (
                    previous_id
                    for previous_id, _count in overlaps
                    if previous_id != identity.session_person_id
                    and previous_id not in assigned_previous
                ),
                key=lambda item: (_live_number(item), item),
            ))
            if absorbed:
                relevant = {identity.session_person_id, *absorbed}
                transitions.append(IdentityTransition(
                    event_type="identity_merged",
                    session_person_id=identity.session_person_id,
                    retained_live_id=identity.session_person_id,
                    absorbed_live_ids=absorbed,
                    overlap_counts=tuple(sorted(
                        (
                            (live_id, count)
                            for live_id, count in overlaps
                            if live_id in relevant
                        ),
                        key=lambda item: (_live_number(item[0]), item[0]),
                    )),
                ))

    active = tuple(sorted(
        (
            (identity.session_person_id, identity.crop_paths)
            for identity in identities
        ),
        key=lambda item: (_live_number(item[0]), item[0]),
    ))
    return IdentityAssignmentResult(
        identities=tuple(identities),
        active_memberships=active,
        transitions=tuple(transitions),
        next_live_number=next_number,
    )
