# Goal 2 Sprint: Automatic Face-Body Pairing

## Scope

- Replace the manual `human_in_the_loop` pairing pause with an automatic single-video matching node.
- Keep the final `build_profile` review interrupt.
- Preserve downstream state keys: `associations`, `frame_groups`, `human_feedback_path`, `quality_body_crops`, and `quality_face_crops`.
- Do not add camera ids, multi-camera memory, global ReID, Docker, or a new face engine.

## Plan

1. Inspect current graph/state/pairing contracts and existing `pairing_feedback.json`.
2. Implement geometry-based face-to-body assignment:
   - face center inside body bbox
   - face inside upper body region
   - face/body height ratio
   - distance to body upper-center
   - detection confidence and crop sharpness
   - Hungarian matching per frame with weak-match rejection
3. Save an auto-pairing feedback JSON compatible with prior feedback shape.
4. Validate auto associations against `pairing_feedback.json` when available.
5. Add simple dominant top/bottom color signals and a clean ReID interface to `profile.json`.
6. Run lightweight syntax/import validation and provide local pipeline test commands.

## Acceptance

- No first human pairing interrupt.
- `promote_crops`, `embed_faces`, `select_best`, `describe_clothing`, `build_profile`, and `finalize` still receive their expected state.
- Profile includes `appearance.color_signals` and `reid` metadata without changing the permanent face embedding contract.
