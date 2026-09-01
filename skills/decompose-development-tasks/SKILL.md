---
name: decompose-development-tasks
description: Use when asked to create, break down, plan, or decompose development tasks.
metadata:
  version: 0.1.0
allowed-tools:
  - get_feature_overview
  - list_tasks
  - create_task
---

Follow this procedure for task decomposition:

1. Inspect the implementation plan, architecture, and existing tasks before
   creating new tasks when the current context is not guaranteed complete.
2. Avoid duplicate tasks when equivalent work already exists.
3. Create atomic, outcome-focused tasks with objective, scope, traceability,
   expected deliverable, verification expectations, and dependencies.
4. Assign exactly one allowed role: `backend_developer`,
   `frontend_developer`, `qa_engineer`, or `code_reviewer`.
5. Never assign tasks to `business_analyst`, `software_architect`, or
   `delivery_manager`.
6. Use the default initial task status.
7. Never claim a task was created until the `create_task` result confirms it.
8. Skills guide behavior only and never grant additional tool access.
