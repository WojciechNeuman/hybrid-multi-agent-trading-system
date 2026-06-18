"""Shared helpers for the notebook *builders* (dev-only — not imported at notebook runtime).

These turn the maintained `.py` modules into self-contained notebook cells so every notebook runs
top-to-bottom in Jupyter with no local imports.
"""
import re
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent          # notebooks_v2/ — where notebooks are WRITTEN
MODDIR = HERE.parent / "mas"                     # src/hmats/mas/ — where modules are now READ from


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "id": uuid.uuid4().hex[:8], "source": source}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def _strip_head(src: str) -> str:
    """Drop the module docstring, the __future__ import, and the __main__ trailer."""
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)
    src = src.replace("from __future__ import annotations\n", "")
    src = re.split(r'\nif __name__ == "__main__":', src)[0]
    return src.strip("\n")


def mas07_engine_source() -> str:
    """The full mas07.py engine as one runnable block (imports + all definitions)."""
    return _strip_head((MODDIR / "mas07.py").read_text())


def mas07_sections():
    """(imports_chunk, [(title, body), ...]) split on mas07.py's section dividers — for notebook 07."""
    src = _strip_head((MODDIR / "mas07.py").read_text())
    parts = re.split(r"# -{10,}\n# (.+?)\n# -{10,}\n", src)
    return parts[0].strip("\n"), [(parts[i], parts[i + 1].strip("\n")) for i in range(1, len(parts), 2)]


def agent_module_source(filename: str) -> str:
    """Body of rule_agents.py / crossasset_agent.py with engine imports stripped (engine is inlined
    separately, so its names — bracket_run, sharpe, OOS_START, ... — are already in scope)."""
    src = _strip_head((MODDIR / filename).read_text())
    src = re.sub(r"\nimport sys\n", "\n", src)               # legacy sys.path hack (pre-package)
    src = re.sub(r"sys\.path\.insert\([^\n]*\)\n", "", src)
    src = re.sub(r"from \.?mas07 import \([^)]*\)\n", "", src)   # multiline import (pkg or bare)
    src = re.sub(r"from \.?mas07 import [^\n]*\n", "", src)      # single-line import (pkg or bare)
    return src.strip("\n")


def helper_source(filename: str) -> str:
    """Body of agent_eval.py / coordinators.py for inlining into notebook 07: strip the engine
    imports and rewrite the ``m.`` (``from . import mas07 as m``) prefix to bare names
    (the engine is already inlined and in scope)."""
    src = _strip_head((MODDIR / filename).read_text())
    src = re.sub(r"from \. import mas07 as m\n", "", src)    # package alias import
    src = re.sub(r"import mas07 as m\n", "", src)            # legacy bare alias (pre-package)
    src = re.sub(r"from \.?mas07 import [^\n]*\n", "", src)  # named import (pkg or bare)
    src = re.sub(r"\bm\.", "", src)
    return src.strip("\n")
