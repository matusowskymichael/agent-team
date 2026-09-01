---
name: write-implementation-plan
description: Use when asked to create, draft, review, save, or record an implementation plan.
metadata:
  version: 0.1.0
allowed-tools:
  - get_feature_overview
  - list_artifacts
  - add_artifact
---

Follow this procedure for implementation plans:

1. Ground the plan in requirements, acceptance criteria, and known
   architecture decisions.
2. Separate confirmed constraints from recommended sequencing and assumptions.
3. Divide work into coherent phases with dependencies, integration points, and
   verification activities.
4. Identify backend, frontend, QA, and code-review responsibilities where
   relevant.
5. Avoid unsupported business behavior and do not include source code.
6. Preview the plan without mutation when the user asks to propose, draft,
   review, or not save.
7. Save only an `implementation_plan` artifact when explicitly requested and
   authorized.
8. Report saved artifact IDs only after a successful tool result.
