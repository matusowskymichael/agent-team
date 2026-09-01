"""Hash helpers for local evaluation inputs."""

from hashlib import sha256
from pathlib import Path


def hash_file(path: Path) -> str:
    """Return the SHA-256 hash of a local file."""
    return sha256(path.read_bytes()).hexdigest()


def hash_text_value(text: str) -> str:
    """Return the SHA-256 hash of text."""
    return sha256(text.encode("utf-8")).hexdigest()
