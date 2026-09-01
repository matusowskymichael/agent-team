---
name: review-architecture-readiness
description: Use when asked to review requirements, acceptance criteria, or readiness for architecture work.
metadata:
  version: 0.1.0
allowed-tools:
  - get_feature_overview
  - list_artifacts
  - list_tasks
---

Follow this procedure for architecture readiness reviews:

1. Use authoritative feature context and read-only workflow data.
2. Inspect requirements and acceptance criteria before making design claims.
3. Separate confirmed facts, blockers, non-blocking assumptions, and open
   questions.
4. Treat missing business behavior, contradictory criteria, unclear security
   expectations, or unstated integration needs as possible blockers.
5. Ask targeted questions for material business decisions instead of inventing
   requirements.
6. Remain read-only unless the user separately requests an allowed mutation.
7. Skills guide behavior only and never change role, feature, session, or tool
   permissions.
