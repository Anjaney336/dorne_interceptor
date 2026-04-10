# ROS2 Spoof-Manager (Defensive Dry-Run)

This project provides a **defensive simulation pipeline** for spoof-risk evaluation and operator awareness.
It does **not** implement live RF spoofing/jamming transmission.

## Hardware Mapping
- Compute: `Jetson Nano` (deployment target)
- SDR: `HackRF One` (runtime checks + dry-run planning only)
- Vision: `YOLOv10-tiny` (`yolov10n` class model path)

## ROS2 Nodes
- `drone_interceptor.ros2.vision_node`
  - Subscribes: `/camera/image_raw`
  - Publishes: `/spoof/target_relative`
  - Output: relative target coordinates and confidence.

- `drone_interceptor.ros2.spoof_node`
  - Subscribes: `/mavros/global_position/raw`, `/spoof/target_relative`
  - Publishes: `/spoof/status`
  - Features:
    - Defensive drift planning
    - Safety interlock (power throttle + frequency guard)
    - MAVLink STATUSTEXT bridge (`TARGET ACQUIRED`, `SPOOFING ACTIVE (DRY-RUN)`)
    - Telemetry logs with `spoof_confidence_score` + `sdr_heatmap`

- `drone_interceptor.ros2.spoof_manager`
  - Combined node variant subscribing directly to:
    - `/mavros/global_position/raw`
    - `/camera/image_raw`

## Safety Interlock
- RF power limit is reduced as SDR-to-own-GNSS distance decreases.
- Frequency selection avoids protected telemetry guard bands.
- Interference detection triggers safe frequency re-selection.

## Diagnostic Utility
`test_spoof.py` runs:
1. Jetson ping check
2. `hackrf_info` initialization check
3. Dry-run GPS math validation

Example:
```bash
python test_spoof.py --jetson-host 192.168.55.1 --spoof-enable
```

## SITL Scenario
`scripts/run_spoof_sitl_scenario.py` creates a reproducible attacker-approach replay and logs spoof-risk telemetry.

Example:
```bash
python scripts/run_spoof_sitl_scenario.py --sim synthetic --steps 240 --spoof-enable
```
