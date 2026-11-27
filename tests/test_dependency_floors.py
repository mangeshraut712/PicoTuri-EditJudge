"""Guard security floors in the root requirements.txt.

GitPython 3.2.1 fixes GHSA-83vg-56qc-22m7 (plus the earlier GHSA-gq48-pqfc-9p58
and GHSA-23mf-xhv8-69c2), so the model-versioning dependency must not
resolve to an older release.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _floor(package):
    pattern = re.compile(rf"^{re.escape(package)}\s*>=\s*([0-9][0-9.]*)", re.I | re.M)
    match = pattern.search((ROOT / "requirements.txt").read_text())
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


class DependencyFloorTests(unittest.TestCase):
    def test_gitpython_floor_includes_security_fix(self):
        floor = _floor("gitpython")
        self.assertIsNotNone(floor, "gitpython must keep a >= floor")
        self.assertGreaterEqual(floor, (3, 2, 1))


if __name__ == "__main__":
    unittest.main()
