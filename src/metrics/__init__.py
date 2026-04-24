"""
DISCERN metrics package.

All metric wrappers follow the same interface:
    compute(candidates: list[str], references: list[str]) -> list[float]

Each module handles its own missing-dependency gracefully (warns and returns
a list of None values) so that callers can proceed with partial results.
"""

from .registry import MetricRegistry, get_registry

__all__ = ["MetricRegistry", "get_registry"]
