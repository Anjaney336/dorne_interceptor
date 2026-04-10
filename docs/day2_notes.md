# Day 2 Notes

This file is the running notes section for Day 2 work. I will keep updating this document for all Day 2 tasks until you explicitly switch the project to Day 3.

## Current Status

- Day 1 baseline was completed and validated:
  - dataset conversion
  - detection
  - tracking
  - simulation
  - end-to-end validation
- Day 2 is now focused on mathematical modeling and control foundations for the interceptor.

## Module 1: Mathematical State Model

### Objective

Define the drone as a discrete-time dynamic system in 2D with acceleration as the control input.

### Continuous-Time Model

The planar motion model is:

- x_dot = v_x
- y_dot = v_y
- v_x_dot = a_x
- v_y_dot = a_y

State vector:

- X = [x, y, v_x, v_y]^T

Control vector:

- U = [a_x, a_y]^T

### Discrete-Time Model

Using timestep `dt`, the model is:

- X[k+1] = A X[k] + B U[k]

Where:

```text
A = [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]

B = [[0.5 dt^2, 0        ],
     [0,        0.5 dt^2],
     [dt,       0        ],
     [0,        dt       ]]
```

### Code Added

- New module: `src/drone_interceptor/dynamics/state_space.py`
- Exported from: `src/drone_interceptor/dynamics/__init__.py`

### Implemented Functions

- `build_state_matrices(dt)`
  - returns discrete-time `A` and `B`
- `update_state(state, acceleration, dt)`
  - updates `[x, y, vx, vy]` by one step

### Example

Initial state:

- X = [0, 0, 1, 2]

Input:

- U = [0.5, -0.2]
- dt = 0.1

Next state:

- x_next = x + vx*dt + 0.5*a_x*dt^2
- y_next = y + vy*dt + 0.5*a_y*dt^2
- vx_next = vx + a_x*dt
- vy_next = vy + a_y*dt

## Verification

- Added tests for:
  - matrix shape and values
  - one-step kinematic update
  - input shape validation

## Module 2: Target Prediction (Core)

### Problem

Estimate where the target will be over the next `N` timesteps.

### Base Physics Model

For each axis:

- x_(t+1) = x_t + v_t * dt + 0.5 * a_t * dt^2

In vector form for planar motion:

- p_(t+1) = p_t + v_t * dt + 0.5 * a_t * dt^2

### Hybrid Upgrade

The Day 2 prediction stack is:

- Physics:
  - handles short-horizon motion reliably
  - guarantees physically consistent rollout
- LSTM residual extension:
  - learns motion patterns from history
  - adds a residual correction on top of the physics rollout

### Input / Output

Input:

- past positions with shape `(T, 2)`

Output:

- predicted trajectory with shape `(N, 2)`
- estimated velocity
- estimated acceleration
- backend label: `physics` or `physics_lstm_hybrid`

### Code Added

- New module: `src/drone_interceptor/prediction/trajectory.py`

### Implemented API

- `HybridTrajectoryPredictor(dt, horizon_steps, history_steps=6, lstm_checkpoint=None, lstm_residual_gain=0.25)`
- `predict(past_positions, acceleration=None, horizon_steps=None)`

### Physics Estimation

From the recent history:

- velocity estimate:
  - v_t = (p_t - p_(t-1)) / dt
- acceleration estimate:
  - a_t = (v_t - v_(t-1)) / dt

If only two positions are available:

- acceleration defaults to zero

## Kalman Filter Layer on Top of the State-Space Model

### Measurement Model

We assume position-only measurements:

- Z_k = H X_k + v_k

Where:

```text
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

### Predict Step

- X_k^- = A X_(k-1) + B U_(k-1)
- P_k^- = A P_(k-1) A^T + Q

### Update Step

- y_k = Z_k - H X_k^-
- S_k = H P_k^- H^T + R
- K_k = P_k^- H^T S_k^-1
- X_k = X_k^- + K_k y_k
- P_k = (I - K_k H) P_k^-

### Code Added

- New module: `src/drone_interceptor/dynamics/kalman.py`

### Implemented API

- `build_observation_matrix()`
- `kalman_predict(state, covariance, acceleration, dt, process_noise)`
- `kalman_update(predicted_state, predicted_covariance, measurement, measurement_noise, observation_matrix=None)`

## Verification

- Added tests for:
  - trajectory rollout from past positions
  - acceleration estimation from history
  - Kalman observation matrix
  - Kalman predict/update dimensions

## Next Likely Day 2 Steps

- extend the model to 3D motion
- fuse the Kalman-estimated state directly into target tracking
- train or plug in a real LSTM residual checkpoint
- connect the predictor output into interception planning constraints

## Module 3: Interception Optimization

### Objective

Minimize interceptor-target separation over time:

- min ||x_i(t) - x_t(t)||

The practical Day 2 implementation targets minimum-time interception by combining:

- Proportional Navigation guidance for fast pursuit
- existing MPC support for receding-horizon control

### Proportional Navigation

The core PN law is:

- a = N * V_c * lambda_dot

Where:

- `N` is the navigation constant
- `V_c` is the closing speed
- `lambda_dot` is the line-of-sight rotation rate

### Day 2 Upgrade for Faster Closure

Pure PN mainly handles line-of-sight rotation. To bias toward faster interception, the implementation adds a time-to-go term based on zero-effort miss:

- time-to-go:
  - t_go = ||r|| / V_c
- zero-effort miss:
  - ZEM = r + v_rel * t_go
- pursuit bias:
  - a_bias = k * ZEM / t_go^2

Final acceleration command:

- a_cmd = a_PN + a_bias

This gives a guidance law that both:

- steers onto the target line-of-sight
- pushes the interceptor toward shorter intercept time

### Input / Output

Input:

- interceptor state
- target state

Output:

- control command with:
  - acceleration command
  - velocity command
  - time-to-go estimate
  - closing speed
  - zero-effort miss

### Code Added

- New module: `src/drone_interceptor/control/guidance.py`

### Implemented API

- `ProportionalNavigationGuidance(dt, navigation_constant=3.0, max_acceleration=18.0, max_speed=55.0, min_closing_speed=0.5, time_to_go_gain=1.0)`
- `compute_command(interceptor_state, target_state)`
- `solve(interceptor_state, target_state)`

### Integration

- The existing control facade in `src/drone_interceptor/control/controller.py` now routes PN mode through the dedicated guidance module.
- MPC remains available as the alternative control backend.

### Verification

- Added tests for:
  - non-zero PN guidance output
  - positive time-to-go and closing speed
  - acceleration clipping under hard limits

## Module 4: Constraint Modeling

### Objective

Make the interceptor mathematically valid and competition-ready by enforcing explicit system constraints.

### Physical Constraints

- maximum velocity:
  - v <= 20 m/s
- acceleration limit:
  - a <= 18 m/s^2

These are explicit in config and enforced in the control layer.

### Tracking Precision Constraint

- allowable tracking error:
  - |e_track| <= 0.5 m

This is represented as a constraint check so the controller can expose whether the current estimate satisfies the precision requirement.

### Drift Constraint

- simulated drift-rate bound:
  - 0.2 <= k <= 0.5 m/s

The navigation layer now clamps configured drift into this admissible band before use.

### Safety Constraint

- no collision
- minimum separation maintained

Implemented with:

- minimum separation:
  - 5.0 m
- collision radius:
  - 1.0 m

When the interceptor enters the minimum-separation zone, the constraint model overrides the raw guidance command with an escape acceleration.

### Code Added

- New module: `src/drone_interceptor/constraints.py`

### Implemented API

- `load_constraint_envelope(config)`
- `clamp_drift_rate(drift_rate_mps, envelope)`
- `tracking_precision_ok(position_error_m, envelope)`
- `ConstraintModel.enforce_guidance_command(...)`

### Integration

- PN guidance now passes through the centralized constraint model.
- Navigation drift is clamped to the admissible range.
- Default config now contains explicit physical, tracking, drift, and safety constraint blocks.

### Verification

- Added tests for:
  - velocity and acceleration clipping
  - safety override inside minimum separation
  - drift clamping
  - tracking precision validation

## Module 5: GPS Drift Model

### Objective

Model gradual navigation perturbation using:

- x_fake = x_true + k t

This captures steadily accumulating bias in the measured position.

### Implemented Model

- scalar drift model:
  - `simulate_axis_drift(x_true, time_s, drift_rate_mps)`
- vector GPS drift model:
  - `simulate_gps_with_drift(true_position, time_s, drift_rate_mps)`

In the current implementation, the drift is injected along the x-axis:

- `[x + k t, y, z]`

### Code

- Updated: `src/drone_interceptor/navigation/state_estimator.py`

## Module 6: Cost Function

### Objective

Define the interception objective:

- J = integral from 0 to T of ( ||x_i - x_t||^2 + alpha ||u||^2 + beta * penalty ) dt

Where:

- distance term:
  - minimizes target separation
- control effort term:
  - minimizes aggressive control
- penalty term:
  - penalizes constraint violations

### Implemented Stage Cost

Discrete-time version:

- J_k = ( ||x_i - x_t||^2 + alpha ||u||^2 + beta * penalty_k ) dt

### Penalty Sources

- velocity clipping
- acceleration clipping
- tracking precision violation
- drift constraint violation
- safety override / collision avoidance activation

### Code Added

- New module: `src/drone_interceptor/optimization/cost.py`

### Implemented API

- `compute_constraint_penalty(constraint_status, penalty_weight=1.0)`
- `compute_interception_cost(interceptor_position, target_position, control_input, dt, alpha=0.1, beta=10.0, constraint_status=None)`
- `InterceptionCostModel.stage_cost(...)`

### Integration

- The autonomy loop now computes and accumulates a scalar stage cost each step.
- Final run summaries now include `total_cost`.

### Verification

- Added tests for:
  - scalar drift model
  - vector GPS drift model
  - constraint penalty calculation
  - combined cost calculation

## Module 7: Optimization Strategy

### Objective

Choose an interceptor trajectory by:

- simulating multiple candidate trajectories
- evaluating the interception cost for each
- selecting the minimum-cost path

### Implemented Strategy

The current Day 2 optimizer is a sampling-based MPC-style rollout optimizer.

It works as follows:

1. generate multiple candidate control sequences
2. propagate interceptor and target dynamics over the horizon
3. evaluate the scalar interception cost for each rollout
4. return the lowest-cost path

This is compatible with later upgrades to:

- gradient descent refinement
- full MPC solvers
- reinforcement learning policies

### Code Added

- New module: `src/drone_interceptor/optimization/trajectory_optimizer.py`

### Implemented API

- `InterceptionTrajectoryOptimizer(config, horizon_steps=None, num_trajectories=None, random_seed=None)`
- `optimize(interceptor_state, target_state, target_acceleration=None)`

### Return Values

The optimizer returns:

- optimal path
- optimal control sequence
- optimal scalar cost
- index of the best candidate
- number of evaluated trajectories

### Verification

- Added tests for:
  - valid optimal path shape
  - valid control sequence shape
  - multiple trajectory evaluation
  - terminal distance reduction for the selected path
