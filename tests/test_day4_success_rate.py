from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.validation.day4 import (
    _apply_day4_tuning,
    _build_config_variant,
    _run_single_scenario,
)


def test_day4_seed12_intercepts_with_validated_terminal_envelope() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    config.setdefault("tracking", {})["mode"] = "kalman"
    config.setdefault("control", {})["mode"] = "mpc"
    _apply_day4_tuning(config)

    scenario = _build_config_variant(config, seed=12)
    result = _run_single_scenario(scenario, use_airsim=False)

    assert result.intercepted is True
