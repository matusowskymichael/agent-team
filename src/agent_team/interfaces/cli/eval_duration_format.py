"""Evaluation duration formatting for CLI output."""


def format_duration(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS, or '-' when unavailable."""
    if seconds is None:
        return "-"
    whole_seconds = max(0, int(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{final_seconds:02}"
