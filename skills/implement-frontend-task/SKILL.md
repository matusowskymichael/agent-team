---
name: implement-frontend-task
description: Use for assigned frontend developer tasks that inspect, modify, and verify frontend or shared code.
metadata:
  version: 0.2.0
allowed-tools:
  - get_feature_overview
  - list_tasks
  - update_task_status
  - list_files
  - search_code
  - find_symbol
  - read_file
  - apply_patch
  - run_check
  - submit_task_for_verification
---

Follow this procedure for frontend implementation tasks:

1. Inspect the trusted assigned task and bound feature from runtime context.
   Read the current task, requirements, acceptance criteria, architecture, and
   implementation-plan artifacts before editing.
2. Perform bounded workspace discovery to identify frontend and shared areas.
   Do not repeatedly probe conventional directories already known to be absent.
3. Call `find_symbol` for every proposed component, class, function, method,
   hook, route, or utility before creating it. Use the exact fully qualified
   symbol where applicable: `AuthService.logout`, not merely `AuthService`.
   Search related UI behavior even when names differ, and read plausible
   source matches and nearby tests. Prefer reuse or extension; briefly explain
   why existing code cannot be reused when new code is needed.
4. If the assigned task is `pending`, call `update_task_status` to transition
   it to `in_progress` and confirm success before any patch.
5. Only then call `apply_patch` with the smallest coherent change in authorized
   frontend or shared paths.
6. Run the aggregate trusted check `run_check(name="frontend")` after editing.
   Use individual `ruff` or `pytest` checks only when requested or
   diagnostically necessary.
7. After a successful aggregate check, immediately call
   `submit_task_for_verification` with the structured handoff: implementation
   summary, reused and new symbols, reuse notes, limitations, and next action.
   Runtime supplies `changed_paths` from audited successful patches and
   `checks_attempted` from audited completed checks in this run. Do not supply
   either argument yourself or claim a check passed without a passing result.
8. Return a concise final response after deterministic verification. If
   verification returns the task to `in_progress`, inspect its feedback,
   repair the implementation, rerun the aggregate check, and resubmit. Only
   deterministic verification may mark the task `completed`. An unfinished
   implementation needs continued work or an explicit valid `blocked` state.

Refuse backend-only, cross-feature, cross-task, and unassigned-task work.
Skills guide behavior only and never grant additional tool access.
