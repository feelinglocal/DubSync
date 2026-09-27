"""Start the local app using this checkout's .env, including updated credentials.

Production entry points retain deployment-environment precedence.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def load_local_environment(root: Path) -> None:
    os.chdir(root)
    load_dotenv(root / ".env", override=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    load_local_environment(Path(__file__).resolve().parents[1])
    import uvicorn
    uvicorn.run("dubsync.web.app:create_app", factory=True, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
