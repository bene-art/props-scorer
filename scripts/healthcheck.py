#!/usr/bin/env python3
"""
Health check script for Docker and CI.

Uses only stdlib (no external dependencies).
Exit 0 on healthy, exit 1 otherwise.
"""

import sys
import urllib.error
import urllib.request

HEALTH_URL = "http://127.0.0.1:8000/health"
TIMEOUT_SECONDS = 5


def main() -> int:
    """Check service health."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=TIMEOUT_SECONDS) as response:
            if response.status == 200:
                return 0
            return 1
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
