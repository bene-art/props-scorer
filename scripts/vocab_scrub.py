#!/usr/bin/env python3
"""Pre-commit gate: block files containing private vocabulary.

The blocklist is stored as salted SHA-256 digests, not plaintext — a
tracked guard file that enumerates the very terms it suppresses would
itself be a leak. Each candidate token in a scanned file is hashed the
same way and compared against the digest set.

Exit 1 (with file:line and a masked preview) on any hit; exit 0 clean.
Usage: vocab_scrub.py FILE [FILE ...]   (pre-commit passes filenames)
"""
from __future__ import annotations

import hashlib
import re
import sys

_SALT = "props-scorer-vocab-v1"

# sha256(salt + lowercased_token) for each blocked term. These are
# one-way digests of the blocklist, not credentials.
_BLOCKED_DIGESTS = {
    "7d5989fc298358d4a40726b2e716ec1465b3a525fafd1c4f102a372b0ca987cc",  # pragma: allowlist secret
    "fe350ec2e6dd759093fb3f677dacdcd2e2ee0c18c4a1db7351e58161d6d8e7f9",  # pragma: allowlist secret
    "23dfb36c73384bb3f2de460858084806ebe83dd4100ce6b51fe2c479ebfe556d",  # pragma: allowlist secret
    "75e803a703e2d13d1d2f3f1dd5b9bb58bf3d46ff7419662cd2bf6a4b4cccddf5",  # pragma: allowlist secret
    "5c2f63e2723a433bb9dbc42d71e29c5c30eeeb902c686a7cb4711cc2a83bc62f",  # pragma: allowlist secret
    "43d0abf1010d146696e2b0f9ed13de1e85735ede4ad099b8969a76887b391ab8",  # pragma: allowlist secret
    "2fe852fa9e2e0ba33c6dd586ba40daa1d76aa21af133778deea1c1b4431462e3",  # pragma: allowlist secret
    "8f19689e6898bc7f625ae759d6987da7b6673f1db39b6509e535b61e10184093",  # pragma: allowlist secret
}

_SCAN_SUFFIXES = (
    ".py", ".md", ".yaml", ".yml", ".toml", ".json", ".jsonl",
    ".sh", ".txt", ".cfg", ".ini",
)

# Words are runs of letters/digits — "com.benai.foo" and "BenAi_local"
# both tokenize so the parts are checked individually.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _digest(token: str) -> str:
    return hashlib.sha256((_SALT + token.lower()).encode()).hexdigest()


def _mask(token: str) -> str:
    return token[0] + "*" * (len(token) - 1)


def scan_file(path: str) -> list[str]:
    hits: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                for token in _TOKEN_RE.findall(line):
                    if _digest(token) in _BLOCKED_DIGESTS:
                        hits.append(f"{path}:{lineno}: blocked term {_mask(token)}")
    except OSError as exc:
        hits.append(f"{path}: unreadable ({exc})")
    return hits


def main(argv: list[str]) -> int:
    hits: list[str] = []
    for path in argv:
        if path.endswith(_SCAN_SUFFIXES) and not path.endswith("vocab_scrub.py"):
            hits.extend(scan_file(path))
    if hits:
        print("Private vocabulary found:")
        print("\n".join(hits))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
