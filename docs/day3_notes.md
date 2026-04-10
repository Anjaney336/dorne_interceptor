# Day 3 Notes

## Objective

Introduce uncertainty modeling, optimal estimation, and robustness-aware control on top of the completed Day 1 and Day 2 stack.

## Modules Implemented

1. Kalman Filter Tracking
   - `src/drone_interceptor/tracking/tracker.py`
   - `TargetTracker` now defaults to a planar Kalman tracker with state `X = [x, y, vx, vy]^T`
   - Explicit Gaussian process noise and measurement noise are modeled

2. Noise Modeling
   - `src/drone_interceptor/perception/detector.py`
   - `src/drone_interceptor/simulation/environment.py`
   - Added synthetic target measurement noise, process noise, and wind disturbance
   - Navigation Kalman fusion exposes `Q/R` scaling through config

3. Prediction Upgrade
   - `src/drone_interceptor/prediction/predictor.py`
   - Target covariance is propagated across the prediction horizon
   - Predicted states now carry uncertainty metadata

4. Control Optimization Upgrade
   - `src/drone_interceptor/optimization/cost.py`
   - `src/drone_interceptor/optimization/trajectory_optimizer.py`
   - `src/drone_interceptor/control/controller.py`
   - Cost upgraded to penalize distance, control effort, constraint violations, and uncertainty
   - MPC mode now uses the uncertainty-aware trajectory optimizer

5. Visualization Upgrade
   - `src/drone_interceptor/visualization/dashboard.py`
   - 3D trajectory visualization now supports measured target path, filtered target path, and interception point

6. Day 3 Validation
   - `src/drone_interceptor/validation/day3.py`
   - `scripts/run_day3_validation.py`
   - Produces:
     - `outputs/day3_3d_simulation.png`
     - `outputs/day3_metrics.png`
     - `logs/day3_validation.log`

## Default Configuration Changes

- Tracking backend defaults to `kalman`
- Control backend defaults to `mpc`
- Added uncertainty weight `optimization.gamma`
- Added process and wind disturbance terms to simulation
- Added synthetic target measurement noise for simulated perception

## Research Framing

Day 3 converts the stack from deterministic interception to a noise-aware cyber-physical model:

`measurement noise -> Kalman filtering -> covariance-aware prediction -> uncertainty-penalized MPC -> robust interception`
