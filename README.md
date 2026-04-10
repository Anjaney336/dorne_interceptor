# AI Drone Interceptor

Research-grade Python scaffold for an AI-based drone interception system. The repository is organized around a modular autonomy stack:

- `perception`: sensing and target detection
- `tracking`: state estimation and target association
- `prediction`: physics-first trajectory prediction with optional LSTM residuals
- `planning`: interception trajectory generation
- `control`: proportional navigation and lightweight MPC
- `navigation`: GPS + IMU fusion with simulated drift
- `simulation`: closed-loop environment and AirSim hooks
- `ros2`: ROS2 and PX4-facing node scaffolds
- `deployment`: Jetson Nano export utilities
- `visualization`: 2D/3D mission dashboards

## Requirements

- Python 3.10+

## Project Layout

```text
.
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── docs/
│   └── architecture.md
├── experiments/
│   └── README.md
├── notebooks/
│   └── README.md
├── requirements.txt
├── pyproject.toml
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── drone_interceptor/
│       ├── config.py
│       ├── logging_utils.py
│       ├── main.py
│       ├── types.py
│       ├── control/
│       ├── perception/
│       ├── planning/
│       ├── prediction/
│       ├── simulation/
│       └── tracking/
└── tests/
    └── test_smoke.py
```

## Virtual Environment Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Pipeline

Use the default YAML configuration:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

The default pipeline now executes an end-to-end autonomous interception loop with:

- Ultralytics YOLO perception for real frames, plus synthetic fallback for simulation
- DeepSORT utilities for video MOT and a kinematic tracker for closed-loop simulation
- Physics-based prediction with an optional LSTM residual checkpoint
- Proportional Navigation or lightweight receding-horizon MPC control
- GPS/IMU Kalman fusion with configurable drift and noise
- Matplotlib mission dashboards and 3D trajectory plots
- ROS2 node entrypoints and PX4 SITL command adapters
- Jetson Nano export helpers for YOLO deployment

Run the full autonomy stack with:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Run the Day 4 physics-based validation stack with:

```bash
python scripts/run_day4_validation.py --project-root .
```

Run the modular platform demo with multi-scenario benchmarking, analytics, and final demo artifacts:

```bash
python scripts/run_platform_demo.py
```

Run the Day 9 DP5 safe simulation package with blueprint video and summary artifacts:

```bash
python scripts/run_day9_execution.py --project-root .
```

Run a seeded autonomy benchmark sweep and export aggregate metrics to JSON with:

```bash
python scripts/run_pipeline.py --config configs/default.yaml --benchmark-runs 5
```

Validate the merged `archive (1)` target dataset + current VisDrone dataset and score candidate YOLOv10 weights with:

```bash
python scripts/run_combined_target_validation.py
```
