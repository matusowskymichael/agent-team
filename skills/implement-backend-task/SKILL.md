---
name: implement-backend-task
description: Use for assigned backend developer tasks that inspect, modify, and verify backend or shared code.
metadata:
  version: 0.1.0
allowed-tools:
  - get_feature_overview
  - list_tasks
  - update_task_status
  - list_files
  - search_code
  - read_file
  - apply_patch
  - run_check
---

Follow this procedure for backend implementation tasks:

1. Confirm the trusted assigned task and bound feature from runtime context.
2. Read the assigned task, requirements, acceptance criteria, architecture, and
   implementation-plan artifacts before editing.
3. List the workspace structure to identify backend and shared areas.
4. Search for proposed class, function, method, endpoint, model, repository,
   service, or utility names before creating them.
5. Search for related existing behavior even when names differ.
6. Read plausible matches and decide whether to reuse, extend, or create.
7. Apply the smallest coherent patch only in authorized backend or shared
   paths.
8. Prefer `run_check(name="backend")` after editing; use individual `ruff`,
   `pyright`, or `pytest` checks only when requested or diagnostically
   necessary.
9. Report changed files, reused implementations, checks, limitations, and any
   workflow status update that succeeded.
10. Refuse frontend-only, cross-feature, cross-task, and unassigned-task work.
11. Skills guide behavior only and never grant additional tool access.
