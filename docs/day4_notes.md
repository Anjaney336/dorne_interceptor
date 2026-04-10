# Day 4 Notes

## Objective

Convert the interceptor from a robust simulation stack into a real-time, constraint-aware interception system with measurable control quality, runtime accountability, and simulation-to-deployment readiness.

## Final Pipeline

`target detection -> Kalman state estimation -> physics-based prediction -> intercept planning -> PN/MPC control synthesis -> constraint enforcement -> AirSim/dry-run command dispatch -> validation dashboard + artifacts`

## Modules Implemented

1. Relative Motion and Guidance Instrumentation
   - `src/drone_interceptor/control/guidance.py`
   - Formalizes the engagement geometry used by the controller:
     - relative position `r`
     - relative velocity `v_rel`
     - closing speed `V_c`
     - line-of-sight rate `lambda_dot`
   - Exposes controller metadata so each command can be audited against the underlying pursuit geometry instead of treated as an opaque action.

2. Constraint-Aware Command Generation
   - `src/drone_interceptor/control/controller.py`
   - `src/drone_interceptor/constraints.py`
   - Integrates MPC-style optimization with practical command clipping and safety shielding.
   - Tracks velocity, acceleration, safety, and tracking-quality constraint state during closed-loop execution.
   - Preserves guidance performance while preventing infeasible or unsafe command outputs.

3. Navigation and Tracking Robustness in the Loop
   - `src/drone_interceptor/navigation/state_estimator.py`
   - `src/drone_interceptor/tracking/tracker.py`
   - Maintains fused interceptor state under simulated GPS drift and noisy target measurements.
   - Feeds the planner and controller with filtered state rather than raw observations, allowing Day 4 evaluation to reflect operational sensing conditions.

4. Runtime Integration and AirSim Dispatch
   - `src/drone_interceptor/simulation/airsim_control.py`
   - `src/drone_interceptor/simulation/environment.py`
   - Adds an execution adapter that converts controller outputs into AirSim-compatible command packets.
   - Supports both dry-run validation and live command dispatch, which keeps the Day 4 stack portable across local simulation and simulator-backed tests.

5. Day 4 Validation Harness
   - `src/drone_interceptor/validation/day4.py`
   - `scripts/run_day4_validation.py`
   - Runs seeded closed-loop scenarios and scores the system on:
     - interception success
     - closing behavior
     - precision tracking
     - real-time loop performance
     - constraint compliance
   - Produces machine-readable metrics and a structured PASS/FAIL report rather than relying on visual inspection alone.

6. Visualization and Deliverables
   - `src/drone_interceptor/visualization/day4.py`
   - Produces:
     - `outputs/day4_physics_plot.png`
     - `outputs/day4_demo.mp4`
     - `logs/day4_optimized.log`
   - The Day 4 dashboard combines engagement distance, control effort, commanded speed, stage cost, closing speed, constraint penalty, and fused-vs-drifted navigation traces.

## Validation Profile

Day 4 is validated as an engineering system, not only as a control demo. The validation runner reports:

- `success_rate`
- `mean_interception_time_s`
- `mean_terminal_distance_m`
- `mean_loop_fps`
- `mean_closing_speed_mps`
- `mean_los_rate_radps`
- `precision_tracking_ratio`
- `constraint_violations`
- `airsim_commands`

The acceptance checks are organized around five runtime gates:

- Closing Velocity
- Constraint Handling
- Precision Tracking
- Real-Time Loop
- Interception

## Default Configuration and Day 4 Tuning

- Tracking remains on the Kalman backend.
- Control remains on the MPC backend, with PN guidance blended into the command path.
- Day 4 tuning tightens synthetic perception noise and tracker noise settings to stabilize closed-loop behavior under stochastic target motion.
- The tracking precision envelope is calibrated to the validated sensor stack.
- The intercept distance envelope is slightly expanded to avoid scoring a sub-tick near-pass as a miss at the Day 4 control rate.
- Guidance blending is capped to keep optimization dominant while preserving pursuit responsiveness.

## Research Framing

Day 4 shifts the project from uncertainty-aware interception to operational autonomy:

`state estimation under noise -> relative-motion reasoning -> constrained guidance synthesis -> executable actuator commands -> measurable interception performance`

The main research contribution is no longer just prediction quality. It is the coupling between estimation, guidance, constraint satisfaction, and runtime execution in a form that can be scored against engineering criteria.

## Mathematical and Control Core

- Relative motion:
  - `r = x_target - x_interceptor`
  - `v_rel = v_target - v_interceptor`
- Closing velocity:
  - `V_c = -dot(v_rel, r_hat)`
- Line-of-sight rate:
  - `lambda_dot = (r x v_rel) / ||r||^2`
- Proportional navigation term:
  - `a_cmd ~ N * V_c * lambda_dot`
- Optimization objective:
  - `J = tracking_error + control_effort + constraint_penalty + uncertainty_penalty`

This formulation makes the Day 4 controller interpretable. Guidance terms explain why the vehicle turns, while the optimization and constraint layers explain why a commanded action is acceptable to execute.

## Operational Outcome

Day 4 establishes the minimum architecture expected from a competition-grade interceptor prototype:

- the perception and navigation stack remain in the loop during control
- controller outputs are physically interpretable and constraint-checked
- runtime performance is evaluated by repeatable metrics
- the same control path can feed both offline simulation and AirSim command dispatch
