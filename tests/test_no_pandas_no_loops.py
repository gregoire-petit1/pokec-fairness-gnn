"""Lint guards: enforce 'no pandas' and 'no Python tensor loops' in src/.

These tests fail fast if any new code imports pandas or reintroduces a Python
loop iterating over individual tensor/dataframe rows in the hot fairness paths.
They are the cheapest enforcement of the user's strict constraints.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PANDAS_PATTERN = re.compile(r"^\s*(import pandas|from pandas)\b", re.MULTILINE)


def _scan_paths_for_pandas() -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for sub in ("src", "tests", "scripts"):
        root = REPO_ROOT / sub
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            text = py.read_text()
            for m in PANDAS_PATTERN.finditer(text):
                line_no = text[: m.start()].count("\n") + 1
                hits.append((py.relative_to(REPO_ROOT), line_no, m.group(0).strip()))
    return hits


@pytest.mark.smoke
def test_no_pandas_imports_anywhere():
    """No `import pandas` or `from pandas` in src/, tests/, scripts/."""
    hits = _scan_paths_for_pandas()
    assert not hits, "pandas imports found:\n" + "\n".join(
        f"  {p}:{ln}  {snippet}" for p, ln, snippet in hits
    )


@pytest.mark.smoke
def test_assortative_mixing_has_no_double_for_loop():
    """The vectorised version uses bincount + reshape, NOT a k×k Python loop."""
    src = (REPO_ROOT / "src" / "fairness" / "metrics.py").read_text()
    # Look for a `for ... in groups` pattern inside assortative_mixing_coefficient
    fn_match = re.search(r"def assortative_mixing_coefficient\b.*?(?=\ndef\s|\Z)", src, re.DOTALL)
    assert fn_match, "assortative_mixing_coefficient not found"
    body = fn_match.group(0)
    # Allow comments mentioning 'for' but no actual for-loop on group keys
    for_loops = re.findall(r"^\s*for\s+\w+\s+in\s+", body, re.MULTILINE)
    assert not for_loops, (
        "assortative_mixing_coefficient should be vectorised; found Python for-loop(s)"
    )


@pytest.mark.smoke
def test_ruff_check_passes():
    """`ruff check src/ tests/ scripts/` exits 0 (no lint errors, no pandas via PD preset)."""
    venv_ruff = REPO_ROOT / ".venv" / "bin" / "ruff"
    ruff_cmd = str(venv_ruff) if venv_ruff.exists() else "ruff"
    result = subprocess.run(
        [ruff_cmd, "check", "src/", "tests/", "scripts/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ruff check failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
