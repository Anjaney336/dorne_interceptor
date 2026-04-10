from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.validation.day1 import validate_integration


def test_validate_integration_passes_when_outputs_exist(tmp_path: Path) -> None:
    required_outputs = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in required_outputs:
        path.write_text("ok", encoding="utf-8")

    statuses = []
    result = validate_integration(statuses=statuses, required_outputs=required_outputs)

    assert result.passed is True


def test_validate_integration_fails_when_output_missing(tmp_path: Path) -> None:
    present = tmp_path / "a.png"
    present.write_text("ok", encoding="utf-8")
    missing = tmp_path / "b.png"

    result = validate_integration(statuses=[], required_outputs=[present, missing])

    assert result.passed is False
