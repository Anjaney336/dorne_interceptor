from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "dashboard" / "app.py"


def _is_port_open(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _build_command(host: str, port: int, headless: bool) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        str(headless).lower(),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Drone Interceptor Streamlit dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Streamlit bind host.")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit bind port.")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser automatically.")
    parser.add_argument("--wait-seconds", type=float, default=25.0, help="Startup wait timeout.")
    args = parser.parse_args()

    if not APP_PATH.exists():
        raise SystemExit(f"Dashboard app not found: {APP_PATH}")

    if _is_port_open(args.host, args.port):
        url = f"http://{args.host}:{args.port}"
        print(f"Dashboard already running at {url}")
        if not args.no_open:
            webbrowser.open(url)
        return

    command = _build_command(host=args.host, port=args.port, headless=False)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = time.time() + max(args.wait_seconds, 1.0)
    while time.time() < deadline:
        if _is_port_open(args.host, args.port):
            url = f"http://{args.host}:{args.port}"
            print(f"Dashboard started at {url} (pid={process.pid})")
            if not args.no_open:
                webbrowser.open(url)
            return
        if process.poll() is not None:
            raise SystemExit("Streamlit process exited before startup completed.")
        time.sleep(0.5)

    raise SystemExit("Timed out waiting for Streamlit to start.")


if __name__ == "__main__":
    main()

