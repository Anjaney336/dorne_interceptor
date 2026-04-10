# Sovereign Interceptor Technical Design Report

**Program:** Honeywell Design-Thon Defense Review  
**System:** Dual-Layer Autonomous Drone Interceptor (State-First Mission Stack)  
**Repository:** `dorne_interceptor`  
**Report Date:** 2026-04-09  
**Prepared By:** Lead Systems Architecture Analysis (Code + Data Grounded)

---

## Document Scope

This report is a code-grounded technical design package using:

- Backend mission logic and telemetry math in:
  - `src/drone_interceptor/simulation/telemetry_api.py`
  - `src/drone_interceptor/simulation/airsim_manager.py`
  - `src/drone_interceptor/navigation/ekf_filter.py`
  - `src/drone_interceptor/ros2/spoof_manager.py`
- Existing simulation datasets in `outputs/`
- Generated evidence bundle:
  - `outputs/sovereign_report/run_mission_summary.csv`
  - `outputs/sovereign_report/per_target_metrics.csv`
  - `outputs/sovereign_report/run_risk_index.csv`
  - `outputs/sovereign_report/notebook_workflow_map.csv`
  - `outputs/sovereign_report/day_progression_metrics.csv`
  - `outputs/sovereign_report/report_stats.json`
  - `outputs/sovereign_report/fig_01 ... fig_14`

---

## Executive Summary

The interceptor stack follows a **state-first architecture** where all UI tables and mission analytics are derived from backend mission state and per-target outputs, not synthetic frontend logic. Multi-target handling is asynchronous (`asyncio.gather`) and each target returns explicit mission metrics (`ekf_success_rate`, `interception_time`, `rmse`, guidance, latency, spoof variance, packet loss, energy, mission success probability).

Empirical data snapshot from run registry (582 mission runs, 3364 per-target rows):

- Mean EKF success rate: **0.4326**
- EKF success P90: **0.85**
- Mean RMSE: **3.906 m**
- RMSE P90: **6.343 m**
- Mean per-target compute latency: **176.892 ms**
- Mean per-target energy consumption: **2355.629 J**
- Threat Risk proxy P90 (run-level): **0.7957**
- Targets tested across runs: **3 to 10**

Interpretation:

- Core architecture is mature and instrumented.
- Performance under stress (high noise/spoof/packet loss) remains the primary optimization axis.
- Day10 benchmark evidence shows high-quality redirection capability in tuned conditions (`redirect_success_rate=1.0`, `tracking_precision_ratio=0.9606`).

---

## Mission Intent and Advanced Methodology (What We Are Doing, How We Did It)

## A. Mission Intent

The system objective is not generic drone tracking. It is a missionized kill-chain with controlled redirection logic:

1. Detect and localize target(s) under sensor noise and telemetry degradation.
2. Maintain robust target state estimates using EKF with innovation-based trust adaptation.
3. Execute interceptor guidance against estimated state while preserving safety constraints.
4. Run spoof-aware defensive logic that distinguishes:
   - adversarial measurement conditions
   - estimator resilience status
   - mission readiness and risk posture
5. Emit complete per-target metrics for post-mission adjudication and scoring.

## B. End-to-End Experimental Protocol

The implemented workflow is reproducible and data-first:

1. Run mission replay engine (`run_replay`) with explicit user payload parameters.
2. Simulate per-frame kinematics, spoof injection toggle, packet-loss model, and latency.
3. Compute frame-level EKF innovations, gates, uncertainty, and terminal geometry.
4. Aggregate target-level metrics:
   - `ekf_success_rate`, `rmse`, `interception_time`, `closest_approach_m`
   - `guidance_efficiency_mps2`, `spoofing_variance`, `compute_latency_ms`
   - `energy_consumption_j`, `mission_success_probability`
5. Aggregate mission-level metrics:
   - target completion, kill-probability ranking, closest approach target
   - quality/deployment gate checks and command readiness
6. Persist all outputs into run registry and figure/CSV artifacts.

## C. Notebook-Integrated Development Workflow

Development was executed as staged notebooks and then hardened into backend services:

- Day 3: probabilistic estimation core (Kalman/EKF workflow)
- Day 4: physics and guidance constraints
- Day 5: full execution baseline and scenario matrix
- Day 6: flight-ready architecture and fallback logic
- Day 8: multi-UAV mission platform architecture
- Day 9: DP5 spoof/redirection upgrade and evidence export
- Day 10: benchmark and edge-readiness evidence packaging

This lineage is now formalized in:

- `outputs/sovereign_report/notebook_workflow_map.csv`
- `outputs/sovereign_report/day_progression_metrics.csv`
- Figures 10 to 14

---

## I. Core Framework and Metric Definitions

## I.1 Mission Stack State Variables

Primary per-target state (backend):

- True state: `x_true = [x, y, z, vx, vy, vz]^T`
- Measurement (possibly spoofed/drifted): `z_meas`
- EKF estimate: `x_hat`
- Innovation: `nu = z_meas - H x_hat^-`
- Uncertainty: `P`
- Spoof metrics: `innovation_m`, `innovation_gate`, `spoof_offset_m`, `spoofing_detected`

Mission-level state:

- `active_stage`, `active_target`, `target_count`
- `rmse_m`, `confidence_score`
- `kill_probability`, `closest_approach_m`
- telemetry reliability (`packet_loss` model outputs)

---

## I.2 RMSE (Root Mean Square Error)

Backend per-target RMSE is computed from frame-wise error:

\[
e_t = \left\| \hat{\mathbf{p}}_t - \mathbf{p}^{true}_t \right\|_2
\]

\[
\text{RMSE} = \sqrt{\frac{1}{T}\sum_{t=1}^{T} e_t^2}
\]

Where:

- \(\hat{\mathbf{p}}_t\): EKF estimated position at frame \(t\)
- \(\mathbf{p}^{true}_t\): true target position at frame \(t\)
- \(T\): number of valid frames for the target

Implementation source: `_compute_target_result_from_replay(...)` in `telemetry_api.py`.

---

## I.3 EKF Success Rate

Per target:

\[
\text{EKF Success Rate} = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}(e_t \le \tau)
\]

Where threshold \(\tau\) is adaptive:

\[
\tau = \text{clip}(0.30 + 0.85\sigma_n + 1.10d_r + 1.50p_l + 0.15r_k,\ 0.45,\ 3.5)
\]

Terms:

- \(\sigma_n\): measurement noise std (m)
- \(d_r\): drift rate (m/s)
- \(p_l\): packet loss rate
- \(r_k\): kill radius (m)

This avoids brittle fixed-threshold scoring under varied mission stress.

---

## I.4 Innovation Gating (Outlier/Spoof Rejection)

Innovation:

\[
\mathbf{\nu}_k = \mathbf{z}_k - H\mathbf{\hat{x}}^-_k
\]

Innovation covariance:

\[
S_k = H P^-_k H^T + R
\]

Mahalanobis distance squared:

\[
D_k^2 = \nu_k^T S_k^{-1}\nu_k
\]

Gate thresholds in code:

- Soft gate: \(\chi^2_{df=3, 0.975}\)
- Hard gate: \(\chi^2_{df=3, 0.995}\)

Decision:

- soft spoof if \(D_k^2 > \chi^2_{0.975}\)
- hard spoof if \(D_k^2 > \chi^2_{0.995}\)

Trust scaling:

- normal: `trust_scale=1`
- soft: `trust_scale=8`
- hard: `trust_scale=64`

This reduces measurement authority instead of unstable hard drop.

---

## I.5 Threat Risk P90

Mission insights build per-target risk index then aggregate run/system percentiles:

\[
R = 0.28R_d + 0.22R_t + 0.20R_f + 0.15R_s + 0.10R_i + 0.05R_p
\]

Where:

- \(R_d\): distance/proximity risk
- \(R_t\): threat-level risk
- \(R_f\): mission-failure risk (\(1-\)engagement success)
- \(R_s\): spoofing risk
- \(R_i\): innovation ratio risk
- \(R_p\): packet risk

Then:

\[
\text{Threat Risk P90} = \text{percentile}_{90}(R_1,\ldots,R_N)
\]

Why P90:

- Captures near-worst operational tail without overreacting to a single outlier.
- Better deployment gate signal than mean risk.

---

## I.6 Drift Rate (\(\Delta V\)) and Soft-Kill Redirection

Drift injection is applied in measurement space:

\[
z_t^{spoof} = x_t^{true} + \Delta x_t + n_t,\quad \Delta x_t = \int_0^t \Delta v(\tau)\,d\tau
\]

In backend replay this is generated through:

- `SpoofingEngine` + `DP5CoordinateSpoofingToolkit`
- `drift_rate_mps` bounded by mission profile
- optional staged attack profiles (`directed`, `linear`, `circular`)

Soft-kill objective: alter perceived coordinates to induce redirection while minimizing broad RF collateral behavior.

---

## I.7 Interception Time

Per target:

\[
T_{int} = t_{impact} - t_{launch}
\]

In code:

- `t_launch` from first frame time
- `t_impact` from first jammed frame or closest-approach timestamp fallback
- `interception_time = max(t_impact - t_launch, 0)`

---

## I.8 Mission Success Probability

Per target mission success probability blends tracking, intercept, distance, and timing:

\[
P_{success} =
0.30P_{ekf}
+0.30P_{track}
+0.25P_{intercept}
+0.10P_{distance}
+0.05P_{time}
\]

With kill-probability kernel based on Mahalanobis distance:

\[
P_{kill}(d_M)=e^{-0.5d_M^2}
\]

---

## II. Mathematical Modeling and Toolchain

## II.1 EKF State Model in 3D

State:

\[
\mathbf{x} =
\begin{bmatrix}
x & y & z & v_x & v_y & v_z
\end{bmatrix}^T
\]

Discrete transition (constant velocity + drift bias):

\[
\mathbf{x}_{k+1} = f(\mathbf{x}_k,\mathbf{u}_k) + \mathbf{w}_k
\]

\[
f(\mathbf{x}_k,\mathbf{u}_k)=
\begin{bmatrix}
x_k + v_{x,k}\Delta t + b_d\Delta t \\
y_k + v_{y,k}\Delta t \\
z_k + v_{z,k}\Delta t \\
v_{x,k} \\
v_{y,k} \\
v_{z,k}
\end{bmatrix}
\]

Measurement:

\[
\mathbf{z}_k = h(\mathbf{x}_k)+\mathbf{v}_k,\quad
h(\mathbf{x}_k)=
\begin{bmatrix}
x_k & y_k & z_k
\end{bmatrix}^T
\]

Jacobian \(F=\frac{\partial f}{\partial x}\):

\[
F=
\begin{bmatrix}
1&0&0&\Delta t&0&0\\
0&1&0&0&\Delta t&0\\
0&0&1&0&0&\Delta t\\
0&0&0&1&0&0\\
0&0&0&0&1&0\\
0&0&0&0&0&1
\end{bmatrix}
\]

Jacobian \(H=\frac{\partial h}{\partial x}\):

\[
H=
\begin{bmatrix}
1&0&0&0&0&0\\
0&1&0&0&0&0\\
0&0&1&0&0&0
\end{bmatrix}
\]

Implementation notes:

- Uses adaptive \(Q\), \(R\) scaling via `adapt_for_tracking_error`.
- Packet loss and drift increase process/measurement uncertainty.

---

## II.2 EKF Covariance Models

Process covariance \(Q\) from acceleration noise:

\[
Q_{axis} =
\begin{bmatrix}
\frac{\Delta t^4}{4}\sigma_a^2 & \frac{\Delta t^3}{2}\sigma_a^2\\
\frac{\Delta t^3}{2}\sigma_a^2 & \Delta t^2 \sigma_a^2
\end{bmatrix}
\]

Measurement covariance \(R\):

\[
R=\text{diag}(\sigma_x^2,\sigma_y^2,\sigma_z^2),\quad \sigma_z \approx 0.67\sigma_{xy}
\]

Adaptive scaling:

- process scale \(\uparrow\) with drift and packet loss
- measurement scale \(\uparrow\) with drift and spoof suspicion

---

## II.3 SNR and Path Loss Physics

Theoretical log-distance path loss:

\[
L(d)=L_0+10\alpha\log_{10}(d/d_0)+X_g
\]

Backend effective SNR approximation:

\[
SNR_{eff} = SNR_{base} - 20\log_{10}(d) - P_{noise} - P_{spoof} - P_{loss} + \epsilon
\]

Where noise/spoof/loss terms are modeled penalties in dB.

---

## II.4 Packet Loss Model (Requested)

Implemented mission model:

\[
PL = 1-\exp\left(-k\cdot \frac{SNR_{inv}}{d^\alpha}\right)
\]

with:

\[
SNR_{inv}=\frac{1}{10^{SNR_{dB}/10}}
\]

Key parameters:

- \(k=0.12\)
- \(\alpha=1.8\)
- distance \(d\): interceptor-target range
- clipped to \([0,0.98]\) and bounded by configured floor

---

## II.5 Guidance and Control

Augmented Proportional Navigation combines:

- LOS-rate normal acceleration
- lead intercept alignment
- pursuit acceleration term
- target acceleration feed-forward

Form:

\[
\mathbf{a}_{cmd}
\approx N V_c (\dot{\lambda}\times \hat{r})
+k_l\mathbf{a}_{lead}
+k_p\mathbf{a}_{pursuit}
+k_t\mathbf{a}_{target}
\]

---

## II.6 PID Acceptance for Spoofed Setpoint (Design Guidance)

To accept controlled spoof-setpoint behavior without triggering target EKF failsafes:

1. Setpoint rate limiting:
   - \(\|\dot{r}_{spoof}\| \le r_{max}\)
2. Gain scheduling:
   - \(K_p, K_i, K_d\) adjusted by innovation confidence
3. Integrator anti-windup:
   - clamp \(I\)-term when innovation gate is exceeded
4. Spoofed setpoint consistency:
   - enforce bounded jerk and bounded heading-rate

Recommended scheduler:

\[
[K_p,K_i,K_d]_{eff} = \gamma(\rho)\,[K_p,K_i,K_d]_{nom},\quad
\rho=\frac{\|\nu\|}{\sqrt{\chi^2_{0.995}}}
\]

with monotone decreasing \(\gamma(\rho)\) for high-risk residuals.

---

## II.7 Energy Model

Per-target dynamic power model in backend:

\[
P(t)=P_{hover}+c_{drag}v(t)^3+\eta\,m\,a(t)\,v(t)
\]

\[
E = \sum_t P(t)\Delta t
\]

Default model constants:

- \(m=6.5\,kg\)
- \(P_{hover}=90\,W\)
- \(c_{drag}=0.02\)
- \(\eta=0.45\)

These are configurable in `/run_mission` payload.

---

## III. Comparative Analysis and Innovation Gap

## III.1 Interception Modality Comparison

| Modality | Strength | Weakness | Friendly-Safety Impact | Precision |
|---|---|---|---|---|
| Kinetic Intercept | Fast neutralization | High collision risk | Medium/High | High terminal, low selective EW |
| Broad RF Jamming | Wide disruption | Affects friendlies, legal risk | High | Low (non-surgical) |
| Directed Energy / Laser | Fast line-of-sight effect | Power + weather constraints | Medium | Medium/High |
| **Directed Drift (this stack)** | Selective GNSS displacement and redirection | Requires estimator rigor and spoof detection discipline | Lower than broad jamming | High when innovation-gated |

---

## III.2 What Is Adopted from PX4/ArduPilot Ecosystem

- MAVLink-style status bridge (`STATUSTEXT` flow in `mavlink_bridge.py`)
- Structured guidance/control loop
- Preflight + mission-state + validation lifecycle
- Separation of flight control, perception, and telemetry services

---

## III.3 Innovation Gap

The architecture moves from brute-force RF denial to **state-consistent displacement**:

- drift profiles tied to kinematics
- innovation gating to prevent implausible jumps
- per-target success/latency/energy accounting
- mission-level readiness and risk gates

---

## III.4 SDR Integration Status

Implemented:

- Defensive dry-run SDR planning interface (`SDRDryRunInterface`)
- runtime checks (`hackrf_info`, gps-sdr-sim presence)
- safety interlock power/frequency gating
- spoof confidence and SDR heatmap logging

Not implemented in this repo:

- live RF transmission chain for offensive waveform generation

This is intentionally constrained for simulation-safe validation.

---

## IV. Constraints and System Architecture

## IV.1 3-Layer Architecture

Layer 1: Flight System (PX4/ArduPilot domain)

- Pixhawk/FCU
- ESC + motors
- true GNSS reference

Layer 2: AI + Vision

- Jetson Nano / Orin Nano
- camera ingest (`/camera/image_raw`)
- YOLOv10-tiny detection pipeline

Layer 3: Spoofing and Defense Analytics

- HackRF One integration checks
- safety interlock
- drift planning + anti-spoofing analytics

---

## IV.2 Hardware-Software Co-Design Considerations

Compute split:

- Jetson GPU: detection inference and model runtime acceleration
- CPU threads: telemetry, EKF updates, ROS2 callbacks, SDR checks

Non-blocking execution:

- Vision and spoof-manager work submitted via thread pools
- heartbeat and mission-state publishing remain independent

---

## IV.3 EMI and Mechanical Constraints (SolidWorks-Oriented)

Recommended placement constraints:

- Jetson at center of gravity with active airflow
- SDR on lower/side arm with shielding
- GNSS antenna on top mast, separated from SDR path
- directional antenna oriented away from own GNSS axis

Required checks:

- SDR-to-own-GNSS distance parameterized in interlock
- frequency guard-bands for 2.4/5.8 GHz telemetry separation

---

## IV.4 Safety Interlock Logic Gates

Power throttle:

\[
P_{limit}(d)=
\begin{cases}
P_{min}, & d \le d_{safe}\\
P_{min} + r(d)\,(P_{max}-P_{min}), & d>d_{safe}
\end{cases}
\]

Frequency interlock:

- detect desired frequency overlap with telemetry guard bands
- auto-select alternate frequency from allowed pool

Mission posture gates from backend insights:

- GREEN: readiness high, risk low
- AMBER: partial readiness, controlled deployment only
- RED: hold/autonomous engagement not authorized

---

## V. Implementation Artifacts and Code Modules

## V.1 EKF Prediction/Assessment Snippet (Python)

```python
def predict(self, drift_rate_mps: float = 0.0, packet_loss: bool = False) -> np.ndarray:
    drift_bias = np.array([[drift_rate_mps * self.dt], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=float)
    self.X = self.A @ self.X + drift_bias
    q_scale = self._adaptive_process_scale * (1.0 + 2.0 * max(float(drift_rate_mps), 0.0)) * (1.15 if packet_loss else 1.0)
    self.P = self.A @ self.P @ self.A.T + self.Q * q_scale
    return self.position
```

```python
def assess(self, z_measured: np.ndarray) -> SpoofingAssessment:
    innovation = z - (self.H @ self.X)
    innovation_covariance = self.H @ self.P @ self.H.T + (self.R * self._adaptive_measurement_scale)
    mahalanobis = float((innovation.T @ np.linalg.inv(innovation_covariance) @ innovation).item())
```

Source: `navigation/ekf_filter.py`.

---

## V.2 FastAPI State-First `/run_mission` Snippet

```python
replay: MissionReplay = await asyncio.to_thread(
    manager.run_replay,
    int(len(target_ids)),
    bool(mission_payload.get("use_ekf", True)),
    float(mission_payload.get("drift_rate_mps", 0.3)),
    ...
)

target_results = await _compute_replay_target_results(
    replay,
    target_ids=target_ids,
    threshold_m=float(ekf_success_threshold_m),
    ...
)
```

```python
mission_results.append(
    {
        "target_id": str(target_id),
        "ekf_success_rate": ekf_success_rate,
        "interception_time": interception_time_s,
        "rmse": rmse_m,
        "guidance_efficiency_mps2": ...,
        "spoofing_variance": ...,
        "compute_latency_ms": ...,
        "energy_consumption_j": ...,
        "mission_success_probability": mission_success_probability,
    }
)
```

Source: `simulation/telemetry_api.py`.

---

## V.3 Jetson Optimization Snippet

```python
exported = model.export(
    format=export_format,
    imgsz=self._imgsz,
    half=half,
    int8=int8,
    optimize=True,
)
```

Source: `deployment/jetson.py` (`JetsonNanoOptimizer.export`).

---

## V.4 ROS2 Spoof-Manager Core Snippet

```python
drift_plan = self._planner.plan(fix, target)
interlock_decision = self._interlock.evaluate(
    sdr_to_own_gnss_distance_m=float(self._config.sdr_to_own_gnss_distance_m),
    desired_frequency_hz=float(self._config.desired_frequency_hz),
    interference_frequency_hz=self._config.telemetry_interference_hz,
)
spoof_confidence = self._spoof_confidence(target, detection_confidence)
```

Source: `ros2/spoof_manager.py`.

---

## V.5 Reproducible Asset Generation

Command:

```bash
python scripts/generate_sovereign_report_assets.py
```

Outputs:

- `outputs/sovereign_report/run_mission_summary.csv`
- `outputs/sovereign_report/per_target_metrics.csv`
- `outputs/sovereign_report/run_risk_index.csv`
- `outputs/sovereign_report/notebook_workflow_map.csv`
- `outputs/sovereign_report/day_progression_metrics.csv`
- `outputs/sovereign_report/report_stats.json`
- `outputs/sovereign_report/fig_01...fig_14.png`

---

## VI. Visual Evidence and Results

## VI.1 Dataset Provenance

Primary source datasets used:

- `outputs/run_registry/run_*.json` (582 complete `run_mission` records)
- `outputs/day9_spoofing_profile.csv` (trajectory spoof profile)
- `outputs/day10_summary.json` (benchmark summary)
- `outputs/judge_sweep_10_runs.csv`
- `outputs/judge_sweep_10_runs_per_target.csv`
- `outputs/judge_sweep_15_runs.csv`
- `notebooks/day*.ipynb` (workflow lineage extraction)
- `outputs/sovereign_report/notebook_workflow_map.csv`
- `outputs/sovereign_report/day_progression_metrics.csv`

---

## VI.2 Aggregate Results Snapshot

From `report_stats.json`:

- `run_registry_mission_runs=582`
- `per_target_samples=3364`
- `targets_range=[3,10]`
- `ekf_success_rate_mean=0.4326`
- `ekf_success_rate_p90=0.85`
- `rmse_mean_m=3.906`
- `rmse_p90_m=6.343`
- `mean_compute_latency_ms=176.892`
- `mean_energy_consumption_j=2355.629`

Spoof mode split (`run_mission_summary.csv`):

- Spoof OFF: EKF success mean `0.4528`, RMSE mean `3.8226`
- Spoof ON: EKF success mean `0.3887`, RMSE mean `4.0878`

Day10 tuned benchmark (`day10_summary.json`):

- Redirect success rate: `1.0`
- Tracking precision ratio: `0.9606`
- Mean tracking error: `0.3375 m`

Notebook workflow map highlights (`notebook_workflow_map.csv`):

- Notebook chain spans Day 3, 4, 5, 6, 8, 9.
- Dominant focus transitions:
  - estimation/tracking -> guidance physics -> architecture -> multi-UAV + spoofing validation.
- Cell-level complexity is highest in Day 3 and Day 9 notebooks, matching algorithmic depth stages.

Day progression highlights (`day_progression_metrics.csv`):

- Day 7 established redirection success baseline (`redirect_success_rate=1.0`).
- Day 9 achieved high tracking precision (`0.95`) and loop throughput (`140.89 FPS`) in tuned profile.
- Day 10 benchmark sustained precision (`0.9606`) with successful redirect criteria.

---

## VI.3 Figure Set

### Figure 1: Packet-Loss Surface

![Packet Loss Surface](../outputs/sovereign_report/fig_01_packet_loss_surface.png)

### Figure 2: EKF Success vs Noise

![EKF Success vs Noise](../outputs/sovereign_report/fig_02_ekf_success_vs_noise.png)

### Figure 3: RMSE Distribution by Spoofing Mode

![RMSE Distribution](../outputs/sovereign_report/fig_03_rmse_distribution_spoof_on_off.png)

### Figure 4: Mahalanobis Probability Curve

![Mahalanobis Probability](../outputs/sovereign_report/fig_04_mahalanobis_probability_curve.png)

### Figure 5: Threat Risk P90 Distribution

![Threat Risk P90 Distribution](../outputs/sovereign_report/fig_05_threat_risk_p90_distribution.png)

### Figure 6: Clean vs Spoofed Trajectory Heatmaps

![Clean vs Spoof Heatmap](../outputs/sovereign_report/fig_06_clean_vs_spoof_heatmap.png)

### Figure 7: Compute Latency Histogram

![Compute Latency Histogram](../outputs/sovereign_report/fig_07_compute_latency_histogram.png)

### Figure 8: Energy vs Interception Time

![Energy vs Interception Time](../outputs/sovereign_report/fig_08_energy_vs_interception_time.png)

### Figure 9: Kill Probability Law

![Kill Probability Curve](../outputs/sovereign_report/fig_09_kill_probability_curve.png)

### Figure 10: Notebook Workflow Timeline

![Notebook Workflow Timeline](../outputs/sovereign_report/fig_10_notebook_workflow_timeline.png)

### Figure 11: Notebook Complexity Profile

![Notebook Complexity Profile](../outputs/sovereign_report/fig_11_notebook_complexity_profile.png)

### Figure 12: Notebook Module Heatmap

![Notebook Module Heatmap](../outputs/sovereign_report/fig_12_notebook_module_heatmap.png)

### Figure 13: Day Progression Dashboard

![Day Progression Dashboard](../outputs/sovereign_report/fig_13_day_progression_dashboard.png)

### Figure 14: Combined Execution Architecture Flow

![Execution Architecture Flow](../outputs/sovereign_report/fig_14_execution_architecture_flow.png)

---

## VII. Mission Readiness Gates for Defense Review

A practical design-thon review gate can be framed as:

1. Estimation:
   - EKF success P90 >= 0.80
   - RMSE P95 <= mission envelope limit
2. Engagement:
   - interception completion >= threshold
   - closest-approach consistency under spoof stress
3. Telemetry:
   - packet-loss observed <= comms envelope
   - latency envelope maintained
4. Safety:
   - interlock pass (power/frequency)
   - non-blocking heartbeat preserved
5. Data integrity:
   - per-target metric completeness = 100%

---

## VIII. Constraints and Assumptions

- The repo currently implements simulation-safe spoof workflow and defensive telemetry analysis.
- Real RF transmission behavior is intentionally not active in this codebase.
- `mission_success` in historical run registry is conservative and can be stricter than per-target interception evidence; this should be interpreted with `interception_completion_rate` and per-target `intercepted` fields together.
- Some legacy runs may reflect earlier tuning phases; trend analyses should be stratified by date/profile for final defense submission.

---

## IX. Required Artifacts for Final Submission Packet

Include the following in presentation handoff:

1. `docs/SOVEREIGN_INTERCEPTOR_TECHNICAL_REPORT.md`
2. `outputs/sovereign_report/*.png`
3. `outputs/sovereign_report/*.csv`
4. `outputs/sovereign_report/report_stats.json`
5. `outputs/day10_summary.json`
6. `outputs/day9_spoofing_profile.csv`

---

## X. Appendix A: Formula Glossary

- RMSE:
  \[
  \sqrt{\frac{1}{T}\sum_t \|\hat{p}_t-p_t\|^2}
  \]
- Innovation:
  \[
  \nu_k = z_k - Hx^-_k
  \]
- Mahalanobis:
  \[
  D_k^2 = \nu_k^T S_k^{-1}\nu_k
  \]
- Kill probability from Mahalanobis:
  \[
  P=\exp(-0.5D^2)
  \]
- Packet loss:
  \[
  PL=1-\exp(-k\cdot SNR_{inv}/d^\alpha)
  \]
- Energy:
  \[
  E=\sum (P_{hover}+c_{drag}v^3+\eta mav)\Delta t
  \]
- Interception time:
  \[
  T_{int}=t_{impact}-t_{launch}
  \]

---

## XI. Appendix B: Toolchain Notes

- NumPy / SciPy: estimator and gating numerics
- Pandas: mission-run and per-target data shaping
- Matplotlib: evidence graph generation
- FastAPI: mission endpoint and state service
- ROS2 node wrappers: spoof-manager + vision + MAVLink status bridge
- Jetson deployment helper: ONNX/TensorRT-ready export path

---

## XII. Appendix C: Recommended Next Iteration

1. Add versioned calibration profiles by environment class (urban, open-field, RF-contested).
2. Expose risk P90 and deployment margin directly in frontend judge tab.
3. Add temporal confidence bands (not only point metrics) for mission probability.
4. Add hardware-in-loop test traces for link SNR and packet loss model validation.
5. Add strict regression tests for metric schema completeness in all tabs.
