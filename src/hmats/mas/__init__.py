"""Multi-agent trading pipeline runtime package.

The engine + coordinator live in :mod:`mas07`; the agent implementations and
evaluation helpers build on it. The notebooks in ``notebooks_v2/`` are generated
by inlining these modules' source (see ``notebooks_v2/_nbinline.py``) and import
nothing at runtime — this package is the importable, single-source-of-truth copy.
"""
from .mas07 import *  # noqa: F401,F403  (engine API: bracket_run, sharpe, OOS_START, ...)
from . import agent_eval, coordinators, crossasset_agent, mas07, rule_agents  # noqa: F401

__all__ = ["mas07", "rule_agents", "crossasset_agent", "agent_eval", "coordinators"]
