"""Minimal file rotation shared by CSV and JSONL logs."""

from __future__ import annotations

import os


def rotate_file(path: str, backup_count: int) -> None:
    """Shift ``path.N`` backups and move the active file to ``path.1``."""
    oldest = f"{path}.{backup_count}"
    if os.path.exists(oldest):
        os.remove(oldest)
    for index in range(backup_count - 1, 0, -1):
        source = f"{path}.{index}"
        if os.path.exists(source):
            os.replace(source, f"{path}.{index + 1}")
    if os.path.exists(path):
        os.replace(path, f"{path}.1")


def exceeds_limit(path: str, pending_bytes: int, max_bytes: int) -> bool:
    return (
        max_bytes > 0
        and os.path.exists(path)
        and os.path.getsize(path) > 0
        and os.path.getsize(path) + pending_bytes > max_bytes
    )
