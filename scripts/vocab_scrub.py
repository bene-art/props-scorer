#!/usr/bin/env python3
"""Pre-commit gate: block files containing private vocabulary.

The blocklist is never stored in this repo. It is read from a local,
untracked file: $VOCAB_SCRUB_TERMS, else ~/.config/vocab-scrub/terms.txt
(one term per line, # comments allowed). Without that file the hook
prints a notice and passes, so clones without the list can still commit.

Exit 1 (with file:line and a masked preview) on any hit; exit 0 clean.
Usage: vocab_scrub.py FILE [FILE ...]   (pre-commit passes filenames)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _load_terms() -> set[str]:
    path = Path(os.environ.get("VOCAB_SCRUB_TERMS", "~/.config/vocab-scrub/terms.txt")).expanduser()
    if not path.is_file():
        return set()
    lines = (line.split("#", 1)[0].strip().lower() for line in path.read_text().splitlines())
    return {term for term in lines if term}


_SCAN_SUFFIXES = (
    ".py", ".md", ".yaml", ".yml", ".toml", ".json", ".jsonl",
    ".sh", ".txt", ".cfg", ".ini",
)

# Words are runs of letters/digits — "com.example.foo" and "Example_local"
# both tokenize so the parts are checked individually.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _mask(token: str) -> str:
    return token[0] + "*" * (len(token) - 1)


def scan_file(path: str, terms: set[str]) -> list[str]:
    hits: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                for token in _TOKEN_RE.findall(line):
                    if token.lower() in terms:
                        hits.append(f"{path}:{lineno}: blocked term {_mask(token)}")
    except OSError as exc:
        hits.append(f"{path}: unreadable ({exc})")
    return hits


def main(argv: list[str]) -> int:
    terms = _load_terms()
    if not terms:
        print("vocab-scrub: no local term list found; skipping (see script docstring)")
        return 0
    hits: list[str] = []
    for path in argv:
        if path.endswith(_SCAN_SUFFIXES) and not path.endswith("vocab_scrub.py"):
            hits.extend(scan_file(path, terms))
    if hits:
        print("Private vocabulary found:")
        print("\n".join(hits))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
