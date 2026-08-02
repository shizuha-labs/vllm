#!/usr/bin/env python3
"""Apply a small source-only unified diff to the pinned vendor runtime.

The base image intentionally contains no git/patch binary.  This strict
applier supports the subset emitted by ``git diff`` and requires every old
hunk to match exactly once, so vendor-source drift fails the image build.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def _apply_hunks(
    source: str, hunks: list[tuple[int, list[str]]], path: str
) -> str:
    source_lines = source.splitlines(keepends=True)
    line_offset = 0
    for index, (old_start, hunk) in enumerate(hunks, start=1):
        old = [line[1:] for line in hunk if line[:1] in (" ", "-")]
        new = [line[1:] for line in hunk if line[:1] in (" ", "+")]
        expected = max(0, old_start - 1 + line_offset)
        candidates = [
            start
            for start in range(0, len(source_lines) - len(old) + 1)
            if source_lines[start : start + len(old)] == old
        ]
        if not candidates:
            raise RuntimeError(f"{path}: hunk {index} did not match vendor source")
        start = min(candidates, key=lambda value: abs(value - expected))
        source_lines[start : start + len(old)] = new
        line_offset += len(new) - len(old)
    return "".join(source_lines)


def apply_patch(root: Path, patch_file: Path) -> list[Path]:
    files: dict[str, list[tuple[int, list[str]]]] = {}
    current_path: str | None = None
    current_hunk: list[str] | None = None
    for line in patch_file.read_text().splitlines(keepends=True):
        if line.startswith("diff --git "):
            current_path = None
            current_hunk = None
        elif line.startswith("+++ b/"):
            current_path = line[6:].strip()
            files.setdefault(current_path, [])
            current_hunk = None
        elif line.startswith("@@ "):
            if current_path is None:
                raise RuntimeError("hunk appeared before target path")
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
            if match is None:
                raise RuntimeError(f"unsupported hunk header: {line.rstrip()}")
            current_hunk = []
            files[current_path].append((int(match.group(1)), current_hunk))
        elif current_hunk is not None and line[:1] in (" ", "+", "-"):
            current_hunk.append(line)

    if not files:
        raise RuntimeError("patch contained no files")
    changed = []
    for relative, hunks in files.items():
        target = root / relative
        if not target.is_file() or not hunks:
            raise RuntimeError(f"invalid patch target: {target}")
        original = target.read_text()
        updated = _apply_hunks(original, hunks, relative)
        target.write_text(updated)
        changed.append(target)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    args = parser.parse_args()
    for path in apply_patch(args.root, args.patch):
        print(f"patched {path}")


if __name__ == "__main__":
    main()
