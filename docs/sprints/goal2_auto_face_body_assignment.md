# Goal 2 v1: Automatic Face-to-Body Assignment

## Objective

Replace the first manual face/body pairing interrupt with a deterministic geometry-only matcher that pairs quality face crops to quality body crops within the same video frame.

## Scope

- Add `auto_associate` as a new person creation node.
- Group `quality_face_crops` and `quality_body_crops` by `video` and `frame_idx`.
- Compute a geometry cost matrix per frame group.
- Use Hungarian assignment to select candidate pairs.
- Accept matches with cost `<= 0.50`.
- Delete unmatched crops and high-cost match crops from disk.
- Preserve the downstream state contract: `associations`, cleaned quality crop lists, and `human_feedback_path`.

## Explicit Non-Goals

- No new models.
- No ReID implementation.
- No model loading changes.
- No changes to `process_video` or `filter_quality`.
- No removal of the second human review interrupt.
- Keep `human_in_the_loop.py` available for rollback.

## Acceptance Criteria

- `filter_quality -> auto_associate -> promote_crops` is the active graph path.
- `pairing_feedback.json` records `confirmed_by_human: false`, `associations`, `deleted_paths`, and `matching_method: "geometry_hungarian_v1"`.
- Logs include frame group count, association count, rejected crop count, and average accepted cost.
- Existing downstream nodes continue to receive the expected crop and association shape.

## Rollback Plan

Restore the graph import and edges to use `human_in_the_loop`:

- import `human_in_the_loop`
- add node `"human_in_the_loop"`
- wire `filter_quality -> human_in_the_loop -> promote_crops`

The old node remains in place for this rollback.
