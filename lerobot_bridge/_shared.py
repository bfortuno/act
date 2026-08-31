"""Import side-effect: put the ACT-fork repo root on ``sys.path``.

The bridge reuses a few standalone modules from the parent repo as the single
source of truth (no vendored copies):

* ``ee_transforms``  - rot6d / relative task-space transforms (numpy + scipy only)
* ``utils``          - ``get_norm_stats`` normalization math (numpy + torch)
* ``ee_sim_env`` / ``sim_env`` / ``constants`` / ``scripted_policy`` - sim eval

Usage::

    import _shared  # noqa: F401  (must precede the imports below)
    import ee_transforms
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
