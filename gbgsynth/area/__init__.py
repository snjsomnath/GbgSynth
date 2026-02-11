"""
Area-level population synthesis package.

This package splits the former monolithic ``area.py`` module into
focused sub-modules while keeping the public API unchanged::

    from gbgsynth.area import GbgArea  # still works

Sub-modules (internal):
    core               — GbgArea orchestrator class
    data_fetcher       — PxWeb API table fetchers
    dwelling_allocator — Dwelling creation, building linking, matching
    exporter           — CSV / DataFrame / format-specific export
    marginal_comparator — Census validation and fit statistics
"""

from gbgsynth.area.core import GbgArea  # noqa: F401
from gbgsynth.api_client import PxWebClient  # noqa: F401  — needed by tests that patch this path

__all__ = ['GbgArea']
