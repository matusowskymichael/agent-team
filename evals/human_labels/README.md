# Human Labels

Human calibration labels are JSONL records with:

- `case_id`
- `rubric_id`
- `rubric_version`
- `scores`
- `verdict`
- `reason`
- `rater`
- `rated_at`

Humans should first review a small sample. Reasons are required, not only
verdicts. Disagreements should improve the rubric, and the judge should be
rechecked whenever the rubric or judge model changes.
