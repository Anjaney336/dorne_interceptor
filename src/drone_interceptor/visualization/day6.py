from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from drone_interceptor.visualization.video import build_video_writers


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_day6_architecture(
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    fallback_points: np.ndarray | None,
    output_path: str | Path,
    title: str = "Day 6 Flight-Ready Architecture",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(14, 10))
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    axis_3d.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], color="#d62728", linewidth=2.2, label="Target")
    axis_3d.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], interceptor_positions[:, 2], color="#1f77b4", linewidth=2.2, label="Interceptor")
    axis_3d.plot(drifted_positions[:, 0], drifted_positions[:, 1], drifted_positions[:, 2], color="#ff7f0e", linewidth=1.8, linestyle="--", label="GNSS Drift")
    axis_3d.plot(fused_positions[:, 0], fused_positions[:, 1], fused_positions[:, 2], color="#2ca02c", linewidth=1.8, linestyle=":", label="EKF Estimate")
    if fallback_points is not None and len(fallback_points) > 0:
        axis_3d.scatter(fallback_points[:, 0], fallback_points[:, 1], fallback_points[:, 2], color="#9467bd", marker="x", s=90, label="Replan Waypoints")
    axis_3d.set_title("3D Mission Geometry")
    axis_3d.set_xlabel("X [m]")
    axis_3d.set_ylabel("Y [m]")
    axis_3d.set_zlabel("Z [m]")
    axis_3d.legend(loc="upper right")

    axis_xy = figure.add_subplot(2, 2, 2)
    axis_xy.plot(target_positions[:, 0], target_positions[:, 1], color="#d62728", linewidth=2.0, label="Target")
    axis_xy.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], color="#1f77b4", linewidth=2.0, label="Interceptor")
    axis_xy.plot(drifted_positions[:, 0], drifted_positions[:, 1], color="#ff7f0e", linewidth=1.8, linestyle="--", label="GNSS Drift")
    axis_xy.plot(fused_positions[:, 0], fused_positions[:, 1], color="#2ca02c", linewidth=1.8, linestyle=":", label="EKF Estimate")
    if fallback_points is not None and len(fallback_points) > 0:
        axis_xy.scatter(fallback_points[:, 0], fallback_points[:, 1], color="#9467bd", s=55, label="Replan")
    axis_xy.set_title("Top-Down Navigation Perturbation")
    axis_xy.set_xlabel("X [m]")
    axis_xy.set_ylabel("Y [m]")
    axis_xy.legend(loc="upper right")

    axis_error = figure.add_subplot(2, 2, 3)
    drift_error = np.linalg.norm(drifted_positions - interceptor_positions, axis=1)
    fusion_error = np.linalg.norm(fused_positions - interceptor_positions, axis=1)
    axis_error.plot(drift_error, color="#ff7f0e", linewidth=2.0, label="Drift Error")
    axis_error.plot(fusion_error, color="#2ca02c", linewidth=2.0, label="EKF Error")
    axis_error.set_title("Navigation Error")
    axis_error.set_xlabel("Step")
    axis_error.set_ylabel("Position Error [m]")
    axis_error.legend(loc="upper left")

    axis_arch = figure.add_subplot(2, 2, 4)
    axis_arch.axis("off")
    architecture_text = "\n".join(
        [
            title,
            "",
            "YOLO -> ROS2 Nodes -> EKF -> PN/MPC -> PX4 SITL -> AirSim",
            "camera -> perception -> tracking -> control -> actuator",
            "GNSS perturbation: x_gps_fake = x_true + k*t",
            "Fallback: waypoint replanning when closure stalls",
            "Edge mode: reduced compute / lower effective FPS",
        ]
    )
    axis_arch.text(0.02, 0.95, architecture_text, va="top", ha="left", fontsize=12, family="monospace")

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def render_day6_demo_video(
    times: np.ndarray,
    target_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    drifted_positions: np.ndarray,
    fused_positions: np.ndarray,
    distances: np.ndarray,
    edge_fps: float,
    node_fps: dict[str, float],
    fallback_flags: np.ndarray,
    output_path: str | Path,
    fps: float = 20.44,
    frame_size: tuple[int, int] = (1280, 720),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    slowdown_factor = 5
    intro_hold_frames = int(round(fps * 1.2))
    outro_hold_frames = int(round(fps * 1.0))

    all_xy = np.vstack([target_positions[:, :2], interceptor_positions[:, :2], drifted_positions[:, :2], fused_positions[:, :2]])
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    span_xy = np.maximum(max_xy - min_xy, 1.0)

    with build_video_writers(output=output, fps=fps, frame_size=frame_size) as writers:
        intro = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        _draw_background(intro)
        _draw_title(intro, "Day 6 Flight Stack Replay", "ROS2-style node pipeline, fallback replanning, and SITL bridge telemetry")
        _draw_intro_card(intro)
        for _ in range(max(intro_hold_frames, 1)):
            writers.write(intro)
        for index in range(max(len(times) - 1, 1)):
            for subframe in range(slowdown_factor):
                alpha = subframe / slowdown_factor
                sample_index = min(index + 1, len(times) - 1)
                blended_time = _lerp_scalar(times[index], times[sample_index], alpha)
                blended_distance = _lerp_scalar(distances[index], distances[sample_index], alpha)
                drift_error = float(
                    np.linalg.norm(
                        _interpolate_state(drifted_positions[index], drifted_positions[sample_index], alpha)
                        - _interpolate_state(fused_positions[index], fused_positions[sample_index], alpha)
                    )
                )
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
                _draw_background(frame)
                _draw_title(frame, "Day 6 Flight Stack Replay", "ROS2-style node pipeline, fallback replanning, and SITL bridge telemetry")
                _draw_stage_banner(frame, blended_time, float(times[-1]))
                _draw_pipeline_panel(frame, node_fps=node_fps, fallback_active=bool(fallback_flags[sample_index]))
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
                    edge_fps=edge_fps,
                    node_fps=node_fps,
                    fallback_active=bool(fallback_flags[sample_index]),
                    drift_error_m=drift_error,
                )
                _draw_backend_logic_card(
                    frame=frame,
                    distance_m=blended_distance,
                    edge_fps=edge_fps,
                    fallback_active=bool(fallback_flags[sample_index]),
                    drift_error_m=drift_error,
                )
                _draw_sparkline(frame, distances[: sample_index + 1], origin=(830, 492), size=(370, 88), color=(82, 190, 255), label="Separation [m]")
                _draw_sparkline(frame, np.linalg.norm(drifted_positions[: sample_index + 1] - interceptor_positions[: sample_index + 1], axis=1), origin=(830, 602), size=(370, 78), color=(255, 174, 82), label="GNSS Drift vs Interceptor [m]")
                writers.write(frame)
        outro = frame.copy() if len(times) > 0 else intro.copy()
        _draw_outcome_card(outro, final_distance_m=float(distances[-1]) if len(distances) > 0 else 0.0)
        for _ in range(max(outro_hold_frames, 1)):
            writers.write(outro)
    return output


def _draw_background(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    top = np.array([8, 14, 22], dtype=float)
    bottom = np.array([14, 26, 36], dtype=float)
    for y in range(height):
        alpha = y / max(height - 1, 1)
        color = ((1.0 - alpha) * top) + (alpha * bottom)
        frame[y, :, :] = np.clip(color, 0, 255).astype(np.uint8)
    for x in range(0, width, 56):
        cv2.line(frame, (x, 0), (x, height), (22, 38, 52), 1, cv2.LINE_AA)
    for y in range(0, height, 56):
        cv2.line(frame, (0, y), (width, y), (22, 38, 52), 1, cv2.LINE_AA)


def _draw_title(frame: np.ndarray, title: str, subtitle: str) -> None:
    cv2.putText(frame, title, (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 246, 248), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (36, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (124, 210, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (32, 92), (1240, 92), (42, 92, 118), 2)


def _draw_intro_card(frame: np.ndarray) -> None:
    cv2.rectangle(frame, (794, 124), (1234, 638), (12, 24, 36), -1)
    cv2.rectangle(frame, (794, 124), (1234, 638), (62, 148, 184), 2)
    lines = [
        "Architecture Replay",
        "Perception publishes detections",
        "Tracking filters noisy measurements",
        "Navigation corrects drift with EKF",
        "Control sends setpoints through the bridge",
        "Fallback replans when closure stalls",
    ]
    for index, line in enumerate(lines):
        scale = 0.74 if index == 0 else 0.58
        weight = 2 if index == 0 else 1
        color = (238, 244, 248) if index == 0 else (186, 224, 238)
        cv2.putText(frame, line, (828, 180 + index * 62), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _draw_stage_banner(frame: np.ndarray, time_s: float, total_time_s: float) -> None:
    progress = 0.0 if total_time_s <= 0.0 else float(np.clip(time_s / total_time_s, 0.0, 1.0))
    if progress < 0.22:
        stage = "PERCEPTION"
    elif progress < 0.45:
        stage = "TRACKING"
    elif progress < 0.70:
        stage = "NAVIGATION + CONTROL"
    else:
        stage = "TERMINAL REPLAN / INTERCEPT"
    cv2.rectangle(frame, (34, 108), (432, 144), (16, 38, 52), -1)
    cv2.rectangle(frame, (34, 108), (432, 144), (80, 198, 246), 1)
    cv2.putText(frame, stage, (46, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (238, 244, 248), 2, cv2.LINE_AA)


def _draw_outcome_card(frame: np.ndarray, final_distance_m: float) -> None:
    cv2.rectangle(frame, (790, 150), (1230, 620), (12, 20, 30), -1)
    cv2.rectangle(frame, (790, 150), (1230, 620), (74, 162, 198), 2)
    lines = [
        "Outcome",
        f"Final separation: {final_distance_m:0.2f} m",
        "This replay shows the flight stack",
        "state path, not a copied mockup.",
    ]
    for index, line in enumerate(lines):
        scale = 0.74 if index == 0 else 0.56
        weight = 2 if index == 0 else 1
        color = (238, 244, 248) if index == 0 else (188, 226, 238)
        cv2.putText(frame, line, (820, 205 + index * 46), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _draw_pipeline_panel(frame: np.ndarray, node_fps: dict[str, float], fallback_active: bool) -> None:
    panel = (36, 158, 398, 676)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (12, 20, 30), -1)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (62, 124, 162), 2)
    cv2.putText(frame, "Node Pipeline", (panel[0] + 18, panel[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (238, 244, 248), 2, cv2.LINE_AA)
    nodes = [
        ("CAMERA", node_fps.get("perception_node", 0.0)),
        ("PERCEPTION", node_fps.get("perception_node", 0.0)),
        ("TRACKING", node_fps.get("tracking_node", 0.0)),
        ("NAVIGATION", node_fps.get("navigation_node", 0.0)),
        ("CONTROL", node_fps.get("control_node", 0.0)),
        ("PX4 / AIRSIM", node_fps.get("control_node", 0.0)),
    ]
    for index, (label, fps_value) in enumerate(nodes):
        y0 = panel[1] + 56 + index * 72
        active_color = (86, 220, 158) if (not fallback_active or index < len(nodes) - 1) else (255, 184, 92)
        cv2.rectangle(frame, (panel[0] + 18, y0), (panel[2] - 18, y0 + 48), (20, 34, 48), -1)
        cv2.rectangle(frame, (panel[0] + 18, y0), (panel[2] - 18, y0 + 48), active_color, 1)
        cv2.putText(frame, label, (panel[0] + 32, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 244, 248), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{fps_value:5.1f} Hz", (panel[0] + 34, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (174, 214, 228), 1, cv2.LINE_AA)
        if index < len(nodes) - 1:
            cv2.arrowedLine(frame, (panel[0] + 180, y0 + 48), (panel[0] + 180, y0 + 66), active_color, 2, cv2.LINE_AA, tipLength=0.35)


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
    panel = (430, 158, 790, 676)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (14, 24, 36), -1)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (62, 124, 162), 2)
    cv2.putText(frame, "Navigation Geometry", (panel[0] + 14, panel[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (238, 244, 248), 2, cv2.LINE_AA)

    for points, color in (
        (target_positions[:, :2], (88, 178, 255)),
        (interceptor_positions[:, :2], (255, 170, 88)),
        (drifted_positions[:, :2], (82, 224, 255)),
        (fused_positions[:, :2], (116, 236, 138)),
    ):
        _draw_polyline(frame, points, panel, min_xy, span_xy, color)
        point = _map_point(points[-1], panel, min_xy, span_xy)
        cv2.circle(frame, point, 6, color, -1)

    legend_items = [
        ("Target", (88, 178, 255)),
        ("Interceptor", (255, 170, 88)),
        ("GNSS Drift", (82, 224, 255)),
        ("EKF Estimate", (116, 236, 138)),
    ]
    for index, (label, color) in enumerate(legend_items):
        y = panel[1] + 54 + index * 28
        cv2.line(frame, (panel[0] + 20, y), (panel[0] + 52, y), color, 3)
        cv2.putText(frame, label, (panel[0] + 62, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 234, 242), 1, cv2.LINE_AA)


def _draw_metrics(
    frame: np.ndarray,
    time_s: float,
    distance_m: float,
    edge_fps: float,
    node_fps: dict[str, float],
    fallback_active: bool,
    drift_error_m: float,
) -> None:
    cv2.putText(frame, "Bridge Telemetry", (826, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (238, 244, 248), 2, cv2.LINE_AA)
    lines = [
        f"t = {time_s:6.2f} s",
        f"distance = {distance_m:7.3f} m",
        f"perception fps = {node_fps.get('perception_node', 0.0):6.2f}",
        f"tracking fps = {node_fps.get('tracking_node', 0.0):6.2f}",
        f"navigation fps = {node_fps.get('navigation_node', 0.0):6.2f}",
        f"control fps = {node_fps.get('control_node', 0.0):6.2f}",
        f"edge mode fps = {edge_fps:6.2f}",
        f"fallback replanner = {'ON' if fallback_active else 'OFF'}",
        f"drift correction delta = {drift_error_m:6.2f} m",
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (826, 174 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (188, 224, 238), 2, cv2.LINE_AA)


def _draw_backend_logic_card(
    frame: np.ndarray,
    distance_m: float,
    edge_fps: float,
    fallback_active: bool,
    drift_error_m: float,
) -> None:
    cv2.rectangle(frame, (820, 154), (1220, 456), (12, 20, 30), -1)
    cv2.rectangle(frame, (820, 154), (1220, 456), (62, 124, 162), 1)
    cv2.putText(frame, "Why The Stack Reacts", (840, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 244, 248), 2, cv2.LINE_AA)
    if fallback_active:
        summary = "Fallback replanning is active because closure stalled or geometry degraded."
    elif distance_m > 120.0:
        summary = "The bridge keeps high-throughput setpoints flowing because the target is still far."
    else:
        summary = "Terminal phase keeps the node chain stable while reducing separation near intercept."
    if drift_error_m > 10.0:
        detail = "Navigation is absorbing significant GNSS drift; EKF keeps the downstream control path coherent."
    elif edge_fps < 30.0:
        detail = "Edge mode is slower, so the pipeline compensates by reusing detections and holding estimates."
    else:
        detail = "All nodes are running fast enough for direct pursuit without emergency replans."
    for index, line in enumerate((summary, detail)):
        cv2.putText(frame, line, (838, 228 + index * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (188, 224, 238), 1, cv2.LINE_AA)


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
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (12, 20, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (58, 114, 146), 1)
    cv2.putText(frame, label, (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 234, 244), 1, cv2.LINE_AA)
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


__all__ = ["plot_day6_architecture", "render_day6_demo_video"]
