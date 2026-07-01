"""Pre-commit guard that blocks committing large data/model binaries.

In an MLOps repo, raw datasets and trained model binaries must be tracked by
DVC (and stored in remote object storage), never committed directly to git.
Accidentally committing a multi-hundred-MB CSV is the single most common and
most painful MLOps mistake. This hook fails the commit if a staged file looks
like a data/model artifact and is not accompanied by a DVC pointer file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Extensions that should always live behind DVC, never in git history.
BLOCKED_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".parquet", ".pkl", ".joblib", ".h5", ".onnx", ".bin", ".pt"}
)
# Files small enough to be config/test fixtures are allowed under these dirs.
ALLOWED_PREFIXES: tuple[str, ...] = ("tests/", "notebooks/")


def is_blocked(path: Path) -> bool:
    """Return True if ``path`` is a data/model artifact that bypasses DVC."""
    posix = path.as_posix()
    if posix.startswith(ALLOWED_PREFIXES):
        return False
    if path.suffix.lower() not in BLOCKED_SUFFIXES:
        return False
    # Allow it only if a sibling DVC pointer exists (file.csv.dvc).
    dvc_pointer = path.with_suffix(path.suffix + ".dvc")
    return not dvc_pointer.exists()


def main(argv: list[str]) -> int:
    """Exit non-zero if any staged file is an untracked data/model artifact."""
    offenders = [arg for arg in argv if is_blocked(Path(arg))]
    if offenders:
        print("[block-untracked-artifacts] Refusing to commit data/model files:")
        for offender in offenders:
            print(f"  - {offender}")
        print(
            "Track these with DVC instead: `dvc add <file>` then commit the "
            ".dvc pointer."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
