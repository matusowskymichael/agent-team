---
name: design-solution-architecture
description: Use when asked to propose, draft, review, save, or explain solution architecture.
metadata:
  version: 0.1.0
allowed-tools:
  - get_feature_overview
  - list_artifacts
  - add_artifact
---

Follow this procedure for solution architecture work:

1. Review authoritative requirements and acceptance criteria first.
2. Identify the requested outcome: review, unsaved proposal, or saved
   architecture artifact.
3. Build traceable recommendations that distinguish business facts,
   assumptions, architectural decisions, tradeoffs, and unresolved questions.
4. Cover only relevant sections such as scope, drivers, components,
   interfaces, data ownership, security, privacy, accessibility, operations,
   testing, risks, and tradeoffs.
5. Do not force irrelevant sections into small features.
6. Preview architecture without mutation when the user asks to propose, draft,
   review, or not save.
7. Save only an `architecture` artifact when explicitly requested and
   authorized.
8. Never write application source code or claim persistence without a
   successful tool result.
