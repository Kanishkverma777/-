"""Tests for the logging setup utility."""

from __future__ import annotations

import logging

from heart_disease.logging_utils import setup_logging


def test_setup_logging() -> None:
    """setup_logging configures root logger handlers without errors."""
    setup_logging(level=logging.DEBUG)
    root = logging.getLogger()
    assert len(root.handlers) > 0
    assert root.level == logging.DEBUG
