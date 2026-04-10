from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from drone_interceptor.visualization.video import build_video_writers


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_day4_dashboard(
    times: np.ndarray,
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    distances: np.ndarray,
    control_effort: np.ndarray,
    commanded_speed: np.ndarray,
    stage_costs: np.ndarray,
    closing_speeds: np.ndarray,
    fps_samples: np.ndarray,
    constraint_penalties: np.ndarray,
    output_path: str | Path,
    intercept_point: np.ndarray | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(16, 10))

    axis_3d = figure.add_subplot(2, 3, 1, projection="3d")
    axis_3d.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], color="#d62728", label="Target", linewidth=2)
    axis_3d.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], interceptor_positions[:, 2], color="#1f77b4", label="Interceptor", linewidth=2)
    if intercept_point is not None:
        intercept = np.asarray(intercept_point, dtype=float)
        axis_3d.scatter(intercept[0], intercept[1], intercept[2], color="#9467bd", marker="x", s=90, label="Intercept")
    axis_3d.set_title("3D Interception Geometry")
    axis_3d.set_xlabel("X [m]")
    axis_3d.set_ylabel("Y [m]")
    axis_3d.set_zlabel("Z [m]")
    axis_3d.legend()

    axis_drift = figure.add_subplot(2, 3, 2)
    axis_drift.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], color="#1f77b4", linewidth=2, label="True Interceptor")
    axis_drift.plot(drifted_positions[:, 0], drifted_positions[:, 1], color="#ff7f0e", linewidth=2, linestyle="--", label="GPS Drift")
    axis_drift.plot(fused_positions[:, 0], fused_positions[:, 1], color="#2ca02c", linewidth=2, linestyle=":", label="Kalman Fusion")
    axis_drift.set_title("GPS Drift Correction")
    axis_drift.set_xlabel("X [m]")
    axis_drift.set_ylabel("Y [m]")
    axis_drift.legend()

    axis_distance = figure.add_subplot(2, 3, 3)
    axis_distance.plot(times, distances, color="#d62728", linewidth=2)
    axis_distance.set_title("Distance vs Time")
    axis_distance.set_xlabel("Time [s]")
    axis_distance.set_ylabel("Distance [m]")

    axis_effort = figure.add_subplot(2, 3, 4)
    axis_effort.plot(times, control_effort, color="#8c564b", linewidth=2, label="Control Effort")
    axis_effort.plot(times, closing_speeds, color="#17becf", linewidth=2, label="Closing Speed")
    axis_effort.set_title("Control and Closure")
    axis_effort.set_xlabel("Time [s]")
    axis_effort.set_ylabel("Magnitude")
    axis_effort.legend()

    axis_velocity = figure.add_subplot(2, 3, 5)
    axis_velocity.plot(times, commanded_speed, color="#9467bd", linewidth=2, label="Commanded Speed")
    axis_velocity.plot(times, fps_samples, color="#2ca02c", linewidth=2, label="Loop FPS")
    axis_velocity.set_title("Velocity and Throughput")
    axis_velocity.set_xlabel("Time [s]")
    axis_velocity.set_ylabel("Value")
    axis_velocity.legend()

    axis_cost = figure.add_subplot(2, 3, 6)
    axis_cost.plot(times, stage_costs, color="#1f77b4", linewidth=2, label="Stage Cost")
    axis_cost.plot(times, constraint_penalties, color="#ff9896", linewidth=2, label="Constraint Penalty")
    axis_cost.set_title("Cost Function")
    axis_cost.set_xlabel("Time [s]")
    axis_cost.set_ylabel("Cost")
    axis_cost.legend()

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def render_day4_demo_video(
    times: np.ndarray,
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    distances: np.ndarray,
    control_effort: np.ndarray,
    commanded_speed: np.ndarray,
    stage_costs: np.ndarray,
    closing_speeds: np.ndarray,
    fps_samples: np.ndarray,
    output_path: str | Path,
    fps: float = 20.44,
    frame_size: tuple[int, int] = (1280, 720),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    slowdown_factor = 5
    intro_hold_frames = int(round(fps * 1.2))
    outro_hold_frames = int(round(fps * 1.0))
    all_xy = np.vstack(
        [
            target_positions[:, :2],
            interceptor_positions[:, :2],
            drifted_positions[:, :2],
            fused_positions[:, :2],
        ]
    )
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)

    with build_video_writers(output, fps=fps, frame_size=frame_size) as writers:
        intro = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        _draw_background(intro)
        _draw_title(intro, "Day 4 Control-Core Replay", "Physics, MPC cost shaping, and drift rejection")
        _draw_intro_card(intro)
        for _ in range(max(intro_hold_frames, 1)):
            writers.write(intro)

        for index in range(max(len(times) - 1, 1)):
            for subframe in range(slowdown_factor):
                alpha = subframe / slowdown_factor
                sample_index = min(index + 1, len(times) - 1)
                blended_time = _lerp_scalar(times[index], times[sample_index], alpha)
                blended_distance = _lerp_scalar(distances[index], distances[sample_index], alpha)
                blended_closing = _lerp_scalar(closing_speeds[index], closing_speeds[sample_index], alpha)
                blended_effort = _lerp_scalar(control_effort[index], control_effort[sample_index], alpha)
                blended_speed = _lerp_scalar(commanded_speed[index], commanded_speed[sample_index], alpha)
                blended_cost = _lerp_scalar(stage_costs[index], stage_costs[sample_index], alpha)
                blended_fps = _lerp_scalar(fps_samples[index], fps_samples[sample_index], alpha)
                fused_sample = _interpolate_state(fused_positions[index], fused_positions[sample_index], alpha)
                drift_sample = _interpolate_state(drifted_positions[index], drifted_positions[sample_index], alpha)
                drift_residual_m = float(np.linalg.norm(drift_sample - fused_sample))
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
                _draw_background(frame)
                _draw_title(frame, "Day 4 Control-Core Replay", "Physics, MPC cost shaping, and drift rejection")
                _draw_stage_banner(frame, time_s=blended_time, total_time_s=float(times[-1]))
                _draw_trajectory_panel(
                    frame=frame,
                    target_positions=np.vstack([target_positions[: index + 1], _interpolate_state(target_positions[index], target_positions[sample_index], alpha)]),
                    interceptor_positions=np.vstack([interceptor_positions[: index + 1], _interpolate_state(interceptor_positions[index], interceptor_positions[sample_index], alpha)]),
                    drifted_positions=np.vstack([drifted_positions[: index + 1], _interpolate_state(drifted_positions[index], drifted_positions[sample_index], alpha)]),
                    fused_positions=np.vstack([fused_positions[: index + 1], _interpolate_state(fused_positions[index], fused_positions[sample_index], alpha)]),
                    min_xy=min_xy,
                    span_xy=span,
                )
                _draw_metric_text(
                    frame=frame,
                    time_s=blended_time,
                    distance_m=blended_distance,
                    closing_speed_mps=blended_closing,
                    control_effort=blended_effort,
                    commanded_speed=blended_speed,
                    stage_cost=blended_cost,
                    loop_fps=blended_fps,
                    drift_residual_m=drift_residual_m,
                )
                _draw_backend_logic_card(
                    frame=frame,
                    distance_m=blended_distance,
                    closing_speed_mps=blended_closing,
                    stage_cost=blended_cost,
                    drift_residual_m=drift_residual_m,
                )
                _draw_sparkline(frame, distances[: sample_index + 1], origin=(820, 370), size=(380, 92), color=(72, 142, 255), label="Separation [m]")
                _draw_sparkline(frame, stage_costs[: sample_index + 1], origin=(820, 490), size=(380, 92), color=(255, 176, 76), label="MPC Stage Cost")
                _draw_sparkline(frame, commanded_speed[: sample_index + 1], origin=(820, 610), size=(380, 72), color=(186, 124, 255), label="Commanded Speed [m/s]")
                writers.write(frame)
        outro = frame.copy() if len(times) > 0 else intro.copy()
        _draw_outcome_card(outro, final_distance_m=float(distances[-1]) if len(distances) > 0 else 0.0)
        for _ in range(max(outro_hold_frames, 1)):
            writers.write(outro)
    return output


def _draw_background(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    top = np.array([16, 26, 42], dtype=float)
    bottom = np.array([7, 12, 20], dtype=float)
    for y in range(height):
        alpha = y / max(height - 1, 1)
        color = ((1.0 - alpha) * top) + (alpha * bottom)
        frame[y, :, :] = np.clip(color, 0, 255).astype(np.uint8)
    for x in range(0, width, 48):
        cv2.line(frame, (x, 0), (x, height), (24, 40, 62), 1, cv2.LINE_AA)
    for y in range(0, height, 48):
        cv2.line(frame, (0, y), (width, y), (24, 40, 62), 1, cv2.LINE_AA)


def _draw_title(frame: np.ndarray, title: str, subtitle: str) -> None:
    cv2.putText(frame, title, (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (238, 244, 248), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (36, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (132, 184, 238), 2, cv2.LINE_AA)
    cv2.line(frame, (32, 92), (1246, 92), (52, 82, 118), 2)


def _draw_intro_card(frame: np.ndarray) -> None:
    cv2.rectangle(frame, (794, 124), (1234, 638), (30, 44, 68), -1)
    cv2.rectangle(frame, (794, 124), (1234, 638), (72, 118, 182), 2)
    lines = [
        "Control-Core Replay",
        "Target and interceptor states propagate in time",
        "Planner drives positive closing speed first",
        "MPC clips commands to stay inside envelope",
        "Drifted GPS is corrected by Kalman fusion",
        "Cost falls as terminal geometry improves",
    ]
    for index, line in enumerate(lines):
        scale = 0.74 if index == 0 else 0.54
        weight = 2 if index == 0 else 1
        color = (240, 246, 248) if index == 0 else (188, 214, 236)
        cv2.putText(frame, line, (824, 180 + index * 62), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _draw_stage_banner(frame: np.ndarray, time_s: float, total_time_s: float) -> None:
    progress = 0.0 if total_time_s <= 0.0 else float(np.clip(time_s / total_time_s, 0.0, 1.0))
    if progress < 0.30:
        stage = "ACQUIRE + TRACK"
    elif progress < 0.68:
        stage = "GUIDANCE + CONTROL"
    else:
        stage = "TERMINAL CLOSURE"
    cv2.rectangle(frame, (34, 108), (338, 142), (28, 52, 82), -1)
    cv2.rectangle(frame, (34, 108), (338, 142), (106, 168, 255), 1)
    cv2.putText(frame, stage, (48, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 242, 248), 2, cv2.LINE_AA)


def _draw_outcome_card(frame: np.ndarray, final_distance_m: float) -> None:
    cv2.rectangle(frame, (790, 150), (1230, 620), (12, 20, 34), -1)
    cv2.rectangle(frame, (790, 150), (1230, 620), (78, 126, 188), 2)
    lines = [
        "Outcome",
        f"Final separation: {final_distance_m:0.2f} m",
        "The interceptor closes because the MPC",
        "keeps velocity feasible while reducing cost.",
    ]
    for index, line in enumerate(lines):
        scale = 0.74 if index == 0 else 0.56
        weight = 2 if index == 0 else 1
        color = (236, 244, 248) if index == 0 else (186, 214, 236)
        cv2.putText(frame, line, (820, 205 + index * 46), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _interpolate_state(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * np.asarray(start, dtype=float)) + (alpha * np.asarray(end, dtype=float))


def _lerp_scalar(start: float, end: float, alpha: float) -> float:
    return float((1.0 - alpha) * float(start) + alpha * float(end))


def _draw_trajectory_panel(
    frame: np.ndarray,
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    min_xy: np.ndarray,
    span_xy: np.ndarray,
) -> None:
    panel = (38, 158, 760, 676)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (36, 58, 88), -1)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (82, 128, 192), 2)
    cv2.putText(frame, "Top-Down Dynamics", (panel[0] + 16, panel[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (238, 244, 248), 2, cv2.LINE_AA)

    _draw_polyline(frame, target_positions[:, :2], panel, min_xy, span_xy, (72, 142, 255))
    _draw_polyline(frame, interceptor_positions[:, :2], panel, min_xy, span_xy, (255, 160, 74))
    _draw_polyline(frame, drifted_positions[:, :2], panel, min_xy, span_xy, (88, 214, 255))
    _draw_polyline(frame, fused_positions[:, :2], panel, min_xy, span_xy, (108, 232, 132))

    for positions, color in (
        (target_positions, (72, 142, 255)),
        (interceptor_positions, (255, 160, 74)),
        (drifted_positions, (88, 214, 255)),
        (fused_positions, (108, 232, 132)),
    ):
        point = _map_point(positions[-1, :2], panel, min_xy, span_xy)
        cv2.circle(frame, point, 6, color, -1)

    legend_items = [
        ("Target", (72, 142, 255)),
        ("Interceptor", (255, 160, 74)),
        ("GPS drift", (88, 214, 255)),
        ("Kalman fusion", (108, 232, 132)),
    ]
    for index, (label, color) in enumerate(legend_items):
        y = panel[1] + 50 + index * 28
        cv2.line(frame, (panel[0] + 18, y), (panel[0] + 50, y), color, 3)
        cv2.putText(frame, label, (panel[0] + 60, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 234, 244), 1, cv2.LINE_AA)


def _draw_metric_text(
    frame: np.ndarray,
    time_s: float,
    distance_m: float,
    closing_speed_mps: float,
    control_effort: float,
    commanded_speed: float,
    stage_cost: float,
    loop_fps: float,
    drift_residual_m: float,
) -> None:
    cv2.putText(frame, "Real-Time Control Metrics", (820, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (236, 244, 248), 2, cv2.LINE_AA)
    lines = [
        f"t = {time_s:6.2f} s",
        f"distance = {distance_m:7.3f} m",
        f"closing speed = {closing_speed_mps:6.3f} m/s",
        f"control effort = {control_effort:6.3f}",
        f"commanded speed = {commanded_speed:6.3f} m/s",
        f"stage cost = {stage_cost:7.3f}",
        f"loop fps = {loop_fps:6.2f}",
        f"drift residual = {drift_residual_m:6.3f} m",
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (820, 164 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (192, 216, 236), 2, cv2.LINE_AA)


def _draw_backend_logic_card(
    frame: np.ndarray,
    distance_m: float,
    closing_speed_mps: float,
    stage_cost: float,
    drift_residual_m: float,
) -> None:
    cv2.rectangle(frame, (808, 152), (1224, 340), (20, 32, 50), -1)
    cv2.rectangle(frame, (808, 152), (1224, 340), (72, 118, 182), 1)
    cv2.putText(frame, "Why The Motion Changes", (826, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 246, 248), 2, cv2.LINE_AA)
    if distance_m > 140.0:
        summary = "Planner biases lead pursuit because the target is still outside terminal range."
    elif closing_speed_mps > 8.0:
        summary = "Positive closure lets MPC keep speed high while respecting the control envelope."
    else:
        summary = "Terminal phase trades speed for precision to avoid overshoot near intercept."
    if drift_residual_m > 6.0:
        detail = "Estimator is actively correcting GPS drift; fused state stays closer to the true path."
    elif stage_cost > 120.0:
        detail = "High stage cost means geometry is still poor, so the controller is reshaping the intercept arc."
    else:
        detail = "Cost is collapsing because closure, heading, and control effort are becoming favorable."
    for index, line in enumerate((summary, detail)):
        cv2.putText(frame, line, (824, 224 + index * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (188, 214, 236), 1, cv2.LINE_AA)


def _draw_sparkline(
    frame: np.ndarray,
    values: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x0, y0 = origin
    width, height = size
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (22, 34, 56), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (62, 98, 146), 1)
    cv2.putText(frame, label, (x0 + 10, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (218, 232, 244), 1, cv2.LINE_AA)
    if len(values) < 2:
        return

    values = np.asarray(values, dtype=float)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    span = max(max_value - min_value, 1e-6)
    x_coords = np.linspace(x0 + 10, x0 + width - 10, len(values))
    y_coords = y0 + height - 10 - ((values - min_value) / span) * (height - 40)
    points = np.stack((x_coords, y_coords), axis=1).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)


def _draw_polyline(
    frame: np.ndarray,
    points_xy: np.ndarray,
    panel: tuple[int, int, int, int],
    min_xy: np.ndarray,
    span_xy: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    if len(points_xy) < 2:
        return
    mapped = np.asarray([_map_point(point, panel, min_xy, span_xy) for point in points_xy], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [mapped], isClosed=False, color=color, thickness=2)


def _map_point(
    point_xy: np.ndarray,
    panel: tuple[int, int, int, int],
    min_xy: np.ndarray,
    span_xy: np.ndarray,
) -> tuple[int, int]:
    x0, y0, x1, y1 = panel
    x_norm = (float(point_xy[0]) - float(min_xy[0])) / float(span_xy[0])
    y_norm = (float(point_xy[1]) - float(min_xy[1])) / float(span_xy[1])
    x = int(x0 + 25 + x_norm * max((x1 - x0) - 50, 1))
    y = int(y1 - 25 - y_norm * max((y1 - y0) - 50, 1))
    return x, y


__all__ = ["plot_day4_dashboard", "render_day4_demo_video"]
