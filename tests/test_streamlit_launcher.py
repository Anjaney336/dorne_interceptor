from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_dashboard_streamlit.py"
SPEC = importlib.util.spec_from_file_location("run_dashboard_streamlit", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_streamlit_launcher_command_shape() -> None:
    cmd = MODULE._build_command(host="127.0.0.1", port=8501, headless=False)
    assert cmd[0]
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4] == str(MODULE.APP_PATH)
    assert "--server.address" in cmd
    assert "--server.port" in cmd
    assert "--server.headless" in cmd


def test_streamlit_launcher_app_path_exists() -> None:
    assert MODULE.APP_PATH.exists()
