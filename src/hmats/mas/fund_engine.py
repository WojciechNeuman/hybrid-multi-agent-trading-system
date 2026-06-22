"""Canonical fund-level multi-agent engine.

This module provides a clearer import name for the final MAS runtime. The legacy
``mas07`` module remains the implementation source for compatibility with the
existing notebooks and artifacts.
"""
from .mas07 import *  # noqa: F401,F403
