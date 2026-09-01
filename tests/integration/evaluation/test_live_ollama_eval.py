"""Live Ollama evaluation smoke tests."""

import os

import pytest

from agent_team.interfaces.cli import eval_cli


@pytest.mark.ollama_eval
@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_EVALS") != "1",
    reason="Set RUN_OLLAMA_EVALS=1 to run live Ollama evals.",
)
class TestLiveOllamaEval:
    """Opt-in live local evaluation behavior tests."""

    @pytest.mark.parametrize(
        "case_id",
        (
            "ba-dev-008",
            "ba-dev-009",
            "ba-dev-013",
            "ba-dev-015",
            "ba-dev-016",
        ),
    )
    def test_selected_business_analyst_cases_with_local_judge(
        self,
        case_id: str,
    ) -> None:
        """Run selected local judged golden cases."""
        exit_code = eval_cli.main(
            [
                "run",
                "--suite",
                "business_analyst_development",
                "--case-id",
                case_id,
                "--candidate-model",
                "qwen3.5:9b",
                "--judge-model",
                "qwen3.8:27b",
                "--repetitions",
                "1",
                "--judge-repetitions",
                "1",
            ],
        )

        assert exit_code in {0, 1}
