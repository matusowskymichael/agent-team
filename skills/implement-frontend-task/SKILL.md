---
name: implement-frontend-task
description: Use for assigned frontend developer tasks that inspect, modify, and verify frontend or shared code.
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

Follow this procedure for frontend implementation tasks:

1. Confirm the trusted assigned task and bound feature from runtime context.
2. Read the assigned task, requirements, acceptance criteria, architecture, and
   implementation-plan artifacts before editing.
3. List the workspace structure to identify frontend and shared areas.
4. Search for proposed component, hook, style, route, and utility names before
   creating them.
5. Search for related existing UI behavior even when names differ.
6. Read plausible matches and decide whether to reuse, extend, or create.
7. Apply the smallest coherent patch only in authorized frontend or shared
   paths.
8. Prefer `run_check(name="frontend")` after editing; use individual `ruff`
   or `pytest` checks only when requested or diagnostically necessary.
9. Report changed files, reused code, checks, limitations, and any workflow
   status update that succeeded.
10. Refuse backend-only, cross-feature, cross-task, and unassigned-task work.
11. Skills guide behavior only and never grant additional tool access.
