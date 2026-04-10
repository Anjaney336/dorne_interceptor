from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from drone_interceptor.visualization.video import build_video_writers


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_day7_spoofing(
    target_positions: np.ndarray,
    drifted_positions: np.ndarray,
    target_estimated_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    baseline_positions: np.ndarray,
    safe_zone: np.ndarray,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(14, 10))
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    axis_3d.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], color="#d62728", linewidth=2.2, label="Target True")
    axis_3d.plot(drifted_positions[:, 0], drifted_positions[:, 1], drifted_positions[:, 2], color="#ff7f0e", linewidth=1.8, linestyle="--", label="Spoofed GPS")
    axis_3d.plot(target_estimated_positions[:, 0], target_estimated_positions[:, 1], target_estimated_positions[:, 2], color="#2ca02c", linewidth=1.8, linestyle=":", label="Target EKF")
    axis_3d.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], interceptor_positions[:, 2], color="#1f77b4", linewidth=2.2, label="Interceptor")
    axis_3d.scatter([safe_zone[0]], [safe_zone[1]], [safe_zone[2]], color="#9467bd", marker="*", s=200, label="Safe Zone")
    axis_3d.set_title("Day 7 Intelligent Drift Geometry")
    axis_3d.set_xlabel("X [m]")
    axis_3d.set_ylabel("Y [m]")
    axis_3d.set_zlabel("Z [m]")
    axis_3d.legend(loc="upper right")

    axis_xy = figure.add_subplot(2, 2, 2)
    axis_xy.plot(baseline_positions[:, 0], baseline_positions[:, 1], color="#7f7f7f", linewidth=1.8, linestyle="-.", label="Baseline Target")
    axis_xy.plot(target_positions[:, 0], target_positions[:, 1], color="#d62728", linewidth=2.0, label="Target True")
    axis_xy.plot(drifted_positions[:, 0], drifted_positions[:, 1], color="#ff7f0e", linewidth=1.8, linestyle="--", label="Spoofed GPS")
    axis_xy.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], color="#1f77b4", linewidth=2.0, label="Interceptor")
    axis_xy.scatter([safe_zone[0]], [safe_zone[1]], color="#9467bd", marker="*", s=160, label="Safe Zone")
    axis_xy.set_title("Top-Down Redirection vs Baseline")
    axis_xy.set_xlabel("X [m]")
    axis_xy.set_ylabel("Y [m]")
    axis_xy.legend(loc="upper right")

    axis_spoof = figure.add_subplot(2, 2, 3)
    spoof_error = np.linalg.norm(drifted_positions - target_positions, axis=1)
    ekf_error = np.linalg.norm(target_estimated_positions - target_positions, axis=1)
    axis_spoof.plot(spoof_error, color="#ff7f0e", linewidth=2.0, label="GPS Spoofing Offset")
    axis_spoof.plot(ekf_error, color="#2ca02c", linewidth=2.0, label="EKF State Error")
    axis_spoof.set_title("Navigation Perturbation Error")
    axis_spoof.set_xlabel("Step")
    axis_spoof.set_ylabel("Error [m]")
    axis_spoof.legend(loc="upper left")

    axis_text = figure.add_subplot(2, 2, 4)
    axis_text.axis("off")
    axis_text.text(
        0.02,
        0.95,
        "\n".join(
            [
                "Day 7 Intelligent Drift",
                "",
                "x_fake(t) = x_true(t) + k(t) * d + eps(t)",
                "k(t): adaptive rate based on interceptor distance",
                "d: mode-dependent direction (linear / circular / safe-zone)",
                "Target EKF consumes spoofed GPS and shifts guidance state",
                "Interceptor remains in closed loop and reacts to target motion",
            ]
        ),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def render_day7_demo_video(
    times: np.ndarray,
    target_positions: np.ndarray,
    drifted_positions: np.ndarray,
    target_estimated_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    baseline_positions: np.ndarray,
    distances: np.ndarray,
    safe_zone_distances: np.ndarray,
    adaptive_rates: np.ndarray,
    spoofing_errors: np.ndarray,
    safe_zone: np.ndarray,
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
            drifted_positions[:, :2],
            target_estimated_positions[:, :2],
            interceptor_positions[:, :2],
            baseline_positions[:, :2],
            safe_zone[:2].reshape(1, 2),
        ]
    )
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    span_xy = np.maximum(max_xy - min_xy, 1.0)

    with build_video_writers(output=output, fps=fps, frame_size=frame_size) as writers:
        intro = np.full((frame_size[1], frame_size[0], 3), 243, dtype=np.uint8)
        cv2.putText(intro, "Day 7 Intelligent Drift Demo", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.line(intro, (30, 58), (1240, 58), (185, 185, 185), 2)
        _draw_intro_card(intro)
        for _ in range(max(intro_hold_frames, 1)):
            writers.write(intro)
        for index in range(max(len(times) - 1, 1)):
            for subframe in range(slowdown_factor):
                alpha = subframe / slowdown_factor
                sample_index = min(index + 1, len(times) - 1)
                frame = np.full((frame_size[1], frame_size[0], 3), 243, dtype=np.uint8)
                cv2.putText(frame, "Day 7 Intelligent Drift Demo", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
                cv2.line(frame, (30, 58), (1240, 58), (185, 185, 185), 2)
                _draw_stage_banner(frame, _lerp_scalar(times[index], times[sample_index], alpha), float(times[-1]))
                _draw_path_panel(
                    frame=frame,
                    target_positions=np.vstack([target_positions[: index + 1], _interpolate_state(target_positions[index], target_positions[sample_index], alpha)]),
                    drifted_positions=np.vstack([drifted_positions[: index + 1], _interpolate_state(drifted_positions[index], drifted_positions[sample_index], alpha)]),
                    target_estimated_positions=np.vstack([target_estimated_positions[: index + 1], _interpolate_state(target_estimated_positions[index], target_estimated_positions[sample_index], alpha)]),
                    interceptor_positions=np.vstack([interceptor_positions[: index + 1], _interpolate_state(interceptor_positions[index], interceptor_positions[sample_index], alpha)]),
                    baseline_positions=baseline_positions[: sample_index + 1],
                    safe_zone=safe_zone,
                    min_xy=min_xy,
                    span_xy=span_xy,
                )
                _draw_day7_metrics(
                    frame=frame,
                    time_s=_lerp_scalar(times[index], times[sample_index], alpha),
                    distance_m=_lerp_scalar(distances[index], distances[sample_index], alpha),
                    safe_zone_distance_m=_lerp_scalar(safe_zone_distances[index], safe_zone_distances[sample_index], alpha),
                    adaptive_rate_mps=_lerp_scalar(adaptive_rates[index], adaptive_rates[sample_index], alpha),
                    spoofing_error_m=_lerp_scalar(spoofing_errors[index], spoofing_errors[sample_index], alpha),
                )
                _draw_sparkline(frame, distances[: sample_index + 1], origin=(820, 355), size=(390, 100), color=(40, 80, 210), label="Interceptor Distance [m]")
                _draw_sparkline(frame, safe_zone_distances[: sample_index + 1], origin=(820, 495), size=(390, 100), color=(150, 80, 190), label="Safe-Zone Distance [m]")
                _draw_sparkline(frame, adaptive_rates[: sample_index + 1], origin=(820, 615), size=(390, 70), color=(30, 160, 70), label="Adaptive Rate [m/s]")
                writers.write(frame)
        outro = frame.copy() if len(times) > 0 else intro.copy()
        _draw_outcome_card(outro, safe_zone_distance_m=float(safe_zone_distances[-1]) if len(safe_zone_distances) > 0 else 0.0)
        for _ in range(max(outro_hold_frames, 1)):
            writers.write(outro)
    return output


def _draw_intro_card(frame: np.ndarray) -> None:
    cv2.rectangle(frame, (800, 120), (1220, 620), (228, 228, 228), 2)
    lines = [
        "Spoofing Replay",
        "Baseline vs redirected path",
        "True target vs spoofed GPS",
        "EKF response to biased nav input",
        "Adaptive drift-rate progression",
        "Safe-zone pull over time",
    ]
    for index, line in enumerate(lines):
        scale = 0.74 if index == 0 else 0.58
        weight = 2 if index == 0 else 1
        color = (28, 28, 28) if index == 0 else (72, 72, 72)
        cv2.putText(frame, line, (826, 180 + index * 62), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _draw_stage_banner(frame: np.ndarray, time_s: float, total_time_s: float) -> None:
    progress = 0.0 if total_time_s <= 0.0 else float(np.clip(time_s / total_time_s, 0.0, 1.0))
    if progress < 0.25:
        stage = "BASELINE TRACK"
    elif progress < 0.50:
        stage = "SPOOFING ACTIVATION"
    elif progress < 0.78:
        stage = "ADAPTIVE DRIFT"
    else:
        stage = "SAFE-ZONE REDIRECTION"
    cv2.rectangle(frame, (32, 66), (388, 98), (232, 238, 244), -1)
    cv2.putText(frame, stage, (44, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (26, 26, 26), 2, cv2.LINE_AA)


def _draw_outcome_card(frame: np.ndarray, safe_zone_distance_m: float) -> None:
    cv2.rectangle(frame, (790, 150), (1230, 620), (248, 248, 248), -1)
    cv2.rectangle(frame, (790, 150), (1230, 620), (210, 210, 210), 2)
    lines = [
        "Outcome",
        f"Final safe-zone distance: {safe_zone_distance_m:0.2f} m",
        "The target path and drift telemetry",
        "come from the backend simulation trace.",
    ]
    for index, line in enumerate(lines):
        scale = 0.74 if index == 0 else 0.56
        weight = 2 if index == 0 else 1
        color = (26, 26, 26) if index == 0 else (70, 70, 70)
        cv2.putText(frame, line, (820, 205 + index * 46), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _interpolate_state(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * np.asarray(start, dtype=float)) + (alpha * np.asarray(end, dtype=float))


def _lerp_scalar(start: float, end: float, alpha: float) -> float:
    return float((1.0 - alpha) * float(start) + alpha * float(end))


def _draw_path_panel(
    frame: np.ndarray,
    target_positions: np.ndarray,
    drifted_positions: np.ndarray,
    target_estimated_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    baseline_positions: np.ndarray,
    safe_zone: np.ndarray,
    min_xy: np.ndarray,
    span_xy: np.ndarray,
) -> None:
    panel = (40, 90, 780, 680)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (210, 210, 210), 2)
    cv2.putText(frame, "Target Redirection Geometry", (panel[0] + 10, panel[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)

    series = [
        (baseline_positions[:, :2], (120, 120, 120)),
        (target_positions[:, :2], (40, 40, 220)),
        (drifted_positions[:, :2], (40, 150, 230)),
        (target_estimated_positions[:, :2], (45, 160, 70)),
        (interceptor_positions[:, :2], (220, 110, 30)),
    ]
    for points, color in series:
        _draw_polyline(frame, points, panel, min_xy, span_xy, color)
        if len(points) > 0:
            cv2.circle(frame, _map_point(points[-1], panel, min_xy, span_xy), 5, color, -1)

    safe_point = _map_point(safe_zone[:2], panel, min_xy, span_xy)
    cv2.drawMarker(frame, safe_point, (160, 70, 190), markerType=cv2.MARKER_STAR, markerSize=18, thickness=2)

    legend = [
        ("Baseline", (120, 120, 120)),
        ("Target True", (40, 40, 220)),
        ("Spoofed GPS", (40, 150, 230)),
        ("Target EKF", (45, 160, 70)),
        ("Interceptor", (220, 110, 30)),
        ("Safe Zone", (160, 70, 190)),
    ]
    for index, (label, color) in enumerate(legend):
        y = panel[1] + 58 + index * 26
        cv2.line(frame, (panel[0] + 18, y), (panel[0] + 48, y), color, 3)
        cv2.putText(frame, label, (panel[0] + 58, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)


def _draw_day7_metrics(
    frame: np.ndarray,
    time_s: float,
    distance_m: float,
    safe_zone_distance_m: float,
    adaptive_rate_mps: float,
    spoofing_error_m: float,
) -> None:
    cv2.putText(frame, "Adaptive Drift Telemetry", (820, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    lines = [
        f"t = {time_s:6.2f} s",
        f"interceptor distance = {distance_m:7.3f} m",
        f"safe-zone distance = {safe_zone_distance_m:7.3f} m",
        f"adaptive drift rate = {adaptive_rate_mps:5.3f} m/s",
        f"spoofing offset = {spoofing_error_m:6.3f} m",
        "target nav = fake GPS + EKF fusion",
        "controller = detection -> tracking -> prediction -> MPC",
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (820, 150 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (55, 55, 55), 2, cv2.LINE_AA)


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
    cv2.putText(frame, label, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 40, 40), 1, cv2.LINE_AA)
    if len(values) < 2:
        return
    values = np.asarray(values, dtype=float)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    span = max(max_value - min_value, 1e-6)
    x_coords = np.linspace(x0 + 10, x0 + width - 10, len(values))
    y_coords = y0 + height - 10 - ((values - min_value) / span) * (height - 35)
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


__all__ = ["plot_day7_spoofing", "render_day7_demo_video"]
