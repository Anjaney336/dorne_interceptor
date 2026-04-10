from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drone_interceptor.platform import run_platform_demo


if __name__ == "__main__":
    run_platform_demo(project_root=ROOT)
