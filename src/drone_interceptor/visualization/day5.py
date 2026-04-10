from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from drone_interceptor.visualization.sensor_gallery import draw_real_sensor_panel
from drone_interceptor.visualization.video import build_video_writers


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_day5_trajectory(
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    output_path: str | Path,
    intercept_point: np.ndarray | None = None,
    title: str = "Day 5 Demo Trajectory",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(1, 1, 1, projection="3d")
    axis.plot(
        target_positions[:, 0],
        target_positions[:, 1],
        target_positions[:, 2],
        color="#d62728",
        linewidth=2.2,
        label="Target",
    )
    axis.plot(
        interceptor_positions[:, 0],
        interceptor_positions[:, 1],
        interceptor_positions[:, 2],
        color="#1f77b4",
        linewidth=2.2,
        label="Interceptor",
    )
    axis.plot(
        drifted_positions[:, 0],
        drifted_positions[:, 1],
        drifted_positions[:, 2],
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        label="Drifted GPS",
    )
    axis.plot(
        fused_positions[:, 0],
        fused_positions[:, 1],
        fused_positions[:, 2],
        color="#2ca02c",
        linewidth=2.0,
        linestyle=":",
        label="Kalman Fusion",
    )
    if intercept_point is not None:
        intercept = np.asarray(intercept_point, dtype=float)
        axis.scatter(
            intercept[0],
            intercept[1],
            intercept[2],
            color="#9467bd",
            marker="x",
            s=100,
            label="Intercept",
        )

    axis.set_title(title)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_day5_distance(
    scenario_names: list[str],
    times_by_scenario: list[np.ndarray],
    distances_by_scenario: list[np.ndarray],
    threshold_m: float,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 7))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    for index, (scenario_name, times, distances) in enumerate(zip(scenario_names, times_by_scenario, distances_by_scenario, strict=False)):
        axis.plot(times, distances, linewidth=2.2, color=palette[index % len(palette)], label=scenario_name)

    axis.axhline(threshold_m, color="#444444", linestyle="--", linewidth=1.5, label=f"Intercept threshold ({threshold_m:.2f} m)")
    axis.set_title("Distance to Target vs Time")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Distance [m]")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def render_day5_demo_video(
    times: np.ndarray,
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    distances: np.ndarray,
    tracking_errors: np.ndarray,
    fps_samples: np.ndarray,
    output_path: str | Path,
    scenario_name: str,
    drift_rate_mps: float,
    fps: float = 20.44,
    frame_size: tuple[int, int] = (1280, 720),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    intro_hold_frames = int(round(fps * 1.5))
    outro_hold_frames = int(round(fps * 1.0))
    slowdown_factor = _compute_playback_multiplier(
        sample_count=len(times),
        fps=fps,
        intro_hold_frames=intro_hold_frames,
        outro_hold_frames=outro_hold_frames,
        min_duration_s=24.0,
        min_multiplier=3,
    )
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
    span_xy = np.maximum(max_xy - min_xy, 1.0)

    with build_video_writers(output=output, fps=fps, frame_size=frame_size) as writers:
        intro = np.full((frame_size[1], frame_size[0], 3), 244, dtype=np.uint8)
        _draw_title(intro, title="Autonomy Stack Mission Replay", subtitle=scenario_name)
        _draw_intro_card(intro, drift_rate_mps=drift_rate_mps, total_duration_s=float(times[-1]) if len(times) > 0 else 0.0)
        draw_real_sensor_panel(intro, panel=(804, 458, 1218, 614), step_index=0, title="Real Perception Reference")
        for _ in range(max(intro_hold_frames, 1)):
            writers.write(intro)

        for index in range(max(len(times) - 1, 1)):
            for subframe in range(slowdown_factor):
                alpha = subframe / slowdown_factor
                sample_index = min(index + 1, len(times) - 1)
                blended_time = _lerp_scalar(times[index], times[sample_index], alpha)
                blended_distance = _lerp_scalar(distances[index], distances[sample_index], alpha)
                blended_tracking = _lerp_scalar(tracking_errors[index], tracking_errors[sample_index], alpha)
                blended_fps = _lerp_scalar(fps_samples[index], fps_samples[sample_index], alpha)
                frame = np.full((frame_size[1], frame_size[0], 3), 245, dtype=np.uint8)
                _draw_title(frame, title="Autonomy Stack Mission Replay", subtitle=scenario_name)
                _draw_stage_banner(frame, time_s=blended_time, total_time_s=float(times[-1]))
                draw_real_sensor_panel(
                    frame,
                    panel=(804, 108, 1218, 262),
                    step_index=sample_index,
                    title="Real Sensor Frames",
                )
                _draw_trajectory_panel(
                    frame=frame,
                    target_positions=np.vstack([target_positions[: index + 1], _interpolate_state(target_positions[index], target_positions[sample_index], alpha)]),
                    interceptor_positions=np.vstack([interceptor_positions[: index + 1], _interpolate_state(interceptor_positions[index], interceptor_positions[sample_index], alpha)]),
                    drifted_positions=np.vstack([drifted_positions[: index + 1], _interpolate_state(drifted_positions[index], drifted_positions[sample_index], alpha)]),
                    fused_positions=np.vstack([fused_positions[: index + 1], _interpolate_state(fused_positions[index], fused_positions[sample_index], alpha)]),
                    min_xy=min_xy,
                    span_xy=span_xy,
                )
                _draw_metrics(
                    frame=frame,
                    time_s=blended_time,
                    distance_m=blended_distance,
                    tracking_error_m=blended_tracking,
                    loop_fps=blended_fps,
                    drift_rate_mps=drift_rate_mps,
                )
                _draw_sparkline(
                    frame=frame,
                    values=distances[: sample_index + 1],
                    origin=(820, 438),
                    size=(390, 84),
                    color=(35, 35, 210),
                    label="Distance [m]",
                )
                _draw_sparkline(
                    frame=frame,
                    values=tracking_errors[: sample_index + 1],
                    origin=(820, 540),
                    size=(390, 84),
                    color=(30, 150, 60),
                    label="Tracking RMSE Proxy [m]",
                )
                _draw_sparkline(
                    frame=frame,
                    values=fps_samples[: sample_index + 1],
                    origin=(820, 642),
                    size=(390, 46),
                    color=(150, 50, 160),
                    label="Loop FPS",
                )
                writers.write(frame)
        outro = frame.copy() if len(times) > 0 else intro.copy()
        _draw_outcome_card(outro, final_distance_m=float(distances[-1]) if len(distances) > 0 else 0.0)
        for _ in range(max(outro_hold_frames, 1)):
            writers.write(outro)
    return output


def _draw_title(frame: np.ndarray, title: str, subtitle: str) -> None:
    cv2.putText(frame, title, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.line(frame, (30, 92), (1240, 92), (185, 185, 185), 2)


def _draw_intro_card(frame: np.ndarray, drift_rate_mps: float, total_duration_s: float) -> None:
    cv2.rectangle(frame, (800, 140), (1220, 620), (225, 225, 225), 2)
    lines = [
        "Storyboard",
        "1. Detection acquires the target",
        "2. Tracking stabilizes the state estimate",
        "3. Interception geometry starts closing",
        "4. Drift is applied to the nav layer",
        "5. The path bends as the estimate shifts",
        "6. The target is redirected toward intercept",
        f"Mission duration: {total_duration_s:0.1f}s",
        f"Configured drift: {drift_rate_mps:0.2f} m/s",
    ]
    for index, line in enumerate(lines):
        scale = 0.72 if index == 0 else 0.56
        weight = 2 if index == 0 else 1
        color = (28, 28, 28) if index == 0 else (68, 68, 68)
        cv2.putText(frame, line, (830, 185 + index * 48), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _draw_stage_banner(frame: np.ndarray, time_s: float, total_time_s: float) -> None:
    progress = 0.0 if total_time_s <= 0.0 else float(np.clip(time_s / total_time_s, 0.0, 1.0))
    if progress < 0.18:
        stage = "DETECTION LOCK"
    elif progress < 0.40:
        stage = "TRACKING + FUSION"
    elif progress < 0.62:
        stage = "TRAJECTORY PREDICTION"
    elif progress < 0.82:
        stage = "DRIFT + GUIDANCE RESPONSE"
    else:
        stage = "TERMINAL INTERCEPT"
    cv2.rectangle(frame, (30, 100), (390, 132), (228, 234, 244), -1)
    cv2.putText(frame, stage, (42, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (26, 26, 26), 2, cv2.LINE_AA)


def _draw_outcome_card(frame: np.ndarray, final_distance_m: float) -> None:
    cv2.rectangle(frame, (780, 150), (1230, 620), (250, 250, 250), -1)
    cv2.rectangle(frame, (780, 150), (1230, 620), (210, 210, 210), 2)
    lines = [
        "Mission Outcome",
        f"Final separation: {final_distance_m:0.2f} m",
        "The replay shows the backend state",
        "changing over time, not a static mockup.",
        "Target path, drifted GPS, fused state,",
        "and interceptor response are all driven",
        "from the simulation trace.",
    ]
    for index, line in enumerate(lines):
        scale = 0.75 if index == 0 else 0.55
        weight = 2 if index == 0 else 1
        color = (26, 26, 26) if index == 0 else (72, 72, 72)
        cv2.putText(frame, line, (810, 205 + index * 50), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _interpolate_state(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * np.asarray(start, dtype=float)) + (alpha * np.asarray(end, dtype=float))


def _lerp_scalar(start: float, end: float, alpha: float) -> float:
    return float((1.0 - alpha) * float(start) + alpha * float(end))


def _compute_playback_multiplier(
    sample_count: int,
    fps: float,
    intro_hold_frames: int,
    outro_hold_frames: int,
    min_duration_s: float,
    min_multiplier: int,
) -> int:
    transitions = max(sample_count - 1, 1)
    target_frames = int(round(float(fps) * float(min_duration_s)))
    narrative_frames = max(target_frames - intro_hold_frames - outro_hold_frames, 0)
    required_multiplier = int(np.ceil(narrative_frames / transitions)) if transitions > 0 else min_multiplier
    return max(int(min_multiplier), required_multiplier)


def _draw_trajectory_panel(
    frame: np.ndarray,
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    min_xy: np.ndarray,
    span_xy: np.ndarray,
) -> None:
    panel = (40, 120, 780, 680)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (210, 210, 210), 2)
    cv2.putText(frame, "Target / Interceptor / Drifted GPS / Fused State", (panel[0] + 10, panel[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    _draw_polyline(frame, target_positions[:, :2], panel, min_xy, span_xy, (40, 40, 220))
    _draw_polyline(frame, interceptor_positions[:, :2], panel, min_xy, span_xy, (220, 110, 30))
    _draw_polyline(frame, drifted_positions[:, :2], panel, min_xy, span_xy, (40, 170, 220))
    _draw_polyline(frame, fused_positions[:, :2], panel, min_xy, span_xy, (40, 150, 50))

    for positions, color in (
        (target_positions, (40, 40, 220)),
        (interceptor_positions, (220, 110, 30)),
        (drifted_positions, (40, 170, 220)),
        (fused_positions, (40, 150, 50)),
    ):
        point = _map_point(positions[-1, :2], panel, min_xy, span_xy)
        cv2.circle(frame, point, 6, color, -1)

    legend_items = [
        ("Target", (40, 40, 220)),
        ("Interceptor", (220, 110, 30)),
        ("Drifted GPS", (40, 170, 220)),
        ("Kalman Fusion", (40, 150, 50)),
    ]
    for index, (label, color) in enumerate(legend_items):
        y = panel[1] + 55 + index * 28
        cv2.line(frame, (panel[0] + 20, y), (panel[0] + 52, y), color, 3)
        cv2.putText(frame, label, (panel[0] + 62, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)


def _draw_metrics(
    frame: np.ndarray,
    time_s: float,
    distance_m: float,
    tracking_error_m: float,
    loop_fps: float,
    drift_rate_mps: float,
) -> None:
    cv2.putText(frame, "Core Metrics", (820, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 30, 30), 2, cv2.LINE_AA)
    lines = [
        f"t = {time_s:6.2f} s",
        f"distance = {distance_m:7.3f} m",
        f"tracking error = {tracking_error_m:6.3f} m",
        f"loop fps = {loop_fps:6.2f}",
        "GNSS: coordinate drift perturbs nav estimation",
        f"drift rate = {drift_rate_mps:4.2f} m/s",
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (820, 326 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (55, 55, 55), 2, cv2.LINE_AA)


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
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (210, 210, 210), 2)
    cv2.putText(frame, label, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
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


__all__ = ["plot_day5_distance", "plot_day5_trajectory", "render_day5_demo_video"]
