from __future__ import annotations

from forensics.person_creation.live_identity import (
    CurrentMembership,
    assign_live_identities,
    maximum_overlap_assignment,
)


def _current(label: int, *paths: str) -> CurrentMembership:
    return CurrentMembership(label, frozenset(paths))


def _ids(result) -> dict[tuple[str, ...], str]:
    return {
        tuple(sorted(item.crop_paths)): item.session_person_id
        for item in result.identities
    }


def test_growth_and_label_change_retain_live_id():
    result = assign_live_identities(
        current_memberships=[_current(91, "a", "b", "c", "d")],
        previous_memberships={"live_0001": frozenset({"a", "b", "c"})},
        next_live_number=2,
    )

    assert result.identities[0].session_person_id == "live_0001"
    assert result.identities[0].cluster_label == 91
    assert result.next_live_number == 2


def test_global_assignment_preserves_second_best_prior_id():
    currents = [
        _current(1, "a1", "a2", "a3", "b1", "b2"),
        _current(2, "a1", "a2", "a3", "a4"),
    ]
    previous = {
        "live_0001": frozenset({"a1", "a2", "a3", "a4"}),
        "live_0002": frozenset({"b1", "b2"}),
    }

    assignment = maximum_overlap_assignment(currents, previous)

    assert {
        current.cluster_label: live_id for current, live_id in assignment
    } == {1: "live_0002", 2: "live_0001"}


def test_input_reordering_does_not_change_assignment():
    currents = [_current(2, "b", "d"), _current(1, "a", "c")]
    previous_a = {
        "live_0002": frozenset({"c", "d"}),
        "live_0001": frozenset({"a", "b"}),
    }
    previous_b = dict(reversed(list(previous_a.items())))

    first = assign_live_identities(
        current_memberships=currents,
        previous_memberships=previous_a,
        next_live_number=3,
    )
    second = assign_live_identities(
        current_memberships=list(reversed(currents)),
        previous_memberships=previous_b,
        next_live_number=3,
    )

    assert _ids(first) == _ids(second)


def test_equal_overlap_ties_use_numeric_id_and_membership_signature_order():
    result = assign_live_identities(
        current_memberships=[_current(8, "b", "d"), _current(9, "a", "c")],
        previous_memberships={
            "live_0002": frozenset({"c", "d"}),
            "live_0001": frozenset({"a", "b"}),
        },
        next_live_number=3,
    )

    assert _ids(result) == {
        ("a", "c"): "live_0001",
        ("b", "d"): "live_0002",
    }


def test_split_retains_parent_and_allocates_child_once():
    result = assign_live_identities(
        current_memberships=[_current(0, "a", "b", "c"), _current(1, "d")],
        previous_memberships={"live_0001": frozenset({"a", "b", "c", "d"})},
        next_live_number=2,
    )
    split = next(item for item in result.transitions if item.event_type == "identity_split")

    assert _ids(result) == {
        ("a", "b", "c"): "live_0001",
        ("d",): "live_0002",
    }
    assert split.previous_live_id == "live_0001"
    assert split.retained_live_id == "live_0001"
    assert split.child_live_ids == ("live_0002",)
    assert dict(split.overlap_counts) == {"live_0001": 3, "live_0002": 1}


def test_merge_uses_oldest_id_on_equal_overlap_and_records_absorbed_id():
    result = assign_live_identities(
        current_memberships=[_current(4, "a", "b", "c", "d")],
        previous_memberships={
            "live_0002": frozenset({"c", "d"}),
            "live_0001": frozenset({"a", "b"}),
        },
        next_live_number=3,
    )
    merge = next(item for item in result.transitions if item.event_type == "identity_merged")

    assert result.identities[0].session_person_id == "live_0001"
    assert merge.retained_live_id == "live_0001"
    assert merge.absorbed_live_ids == ("live_0002",)
    assert dict(merge.overlap_counts) == {"live_0001": 2, "live_0002": 2}


def test_disappearance_reappearance_and_monotonic_non_reuse():
    disappeared = assign_live_identities(
        current_memberships=[],
        previous_memberships={"live_0001": frozenset({"a"})},
        next_live_number=2,
    )
    reappeared = assign_live_identities(
        current_memberships=[_current(0, "new-crop")],
        previous_memberships=dict(disappeared.active_memberships),
        next_live_number=disappeared.next_live_number,
    )
    later = assign_live_identities(
        current_memberships=[_current(1, "another-crop")],
        previous_memberships={},
        next_live_number=reappeared.next_live_number,
    )

    assert disappeared.active_memberships == ()
    assert reappeared.identities[0].session_person_id == "live_0002"
    assert later.identities[0].session_person_id == "live_0003"


def test_simultaneous_split_and_merge_describe_final_assignment():
    result = assign_live_identities(
        current_memberships=[
            _current(0, "a1", "a2", "b1"),
            _current(1, "a3"),
        ],
        previous_memberships={
            "live_0001": frozenset({"a1", "a2", "a3"}),
            "live_0002": frozenset({"b1"}),
        },
        next_live_number=3,
    )
    split = next(item for item in result.transitions if item.event_type == "identity_split")
    merge = next(item for item in result.transitions if item.event_type == "identity_merged")

    assert _ids(result) == {
        ("a1", "a2", "b1"): "live_0001",
        ("a3",): "live_0003",
    }
    assert split.retained_live_id == "live_0001"
    assert split.child_live_ids == ("live_0003",)
    assert dict(split.overlap_counts) == {"live_0001": 2, "live_0003": 1}
    assert merge.retained_live_id == "live_0001"
    assert merge.absorbed_live_ids == ("live_0002",)
    assert dict(merge.overlap_counts) == {"live_0001": 2, "live_0002": 1}


def test_raw_many_to_many_overlap_emits_no_empty_transition_events():
    currents = [_current(0, "a", "c"), _current(1, "b", "d")]
    previous = {
        "live_0001": frozenset({"a", "b"}),
        "live_0002": frozenset({"c", "d"}),
    }

    first = assign_live_identities(
        current_memberships=currents,
        previous_memberships=previous,
        next_live_number=3,
    )
    second = assign_live_identities(
        current_memberships=list(reversed(currents)),
        previous_memberships=dict(reversed(list(previous.items()))),
        next_live_number=3,
    )

    assert _ids(first) == _ids(second)
    assert not any(
        item.event_type in {"identity_split", "identity_merged"}
        for item in first.transitions
    )
    assert first.transitions == second.transitions
