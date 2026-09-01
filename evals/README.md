# Agent Team Local Evaluations

This directory contains manually maintained golden datasets and strict
rubrics for local evaluation of prompt, model, tool, and harness changes.

Run development cases first, inspect failures, change one behavior at a time,
then compare results. Use holdout cases sparingly after development results
improve. Never rewrite golden expectations merely to make a failing run pass.

Generated results are written to `.agent_team/evals/`.
