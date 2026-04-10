from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from drone_interceptor.visualization.video import build_video_writers


def render_day9_demo_video(
    times: np.ndarray,
    true_positions: np.ndarray,
    spoofed_positions: np.ndarray,
    estimated_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    safe_zone: np.ndarray,
    drift_rates: np.ndarray,
    tracking_errors: np.ndarray,
    output_path: str | Path,
    fps: float = 20.44,
    frame_size: tuple[int, int] = (1280, 720),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    intro_hold_frames = int(round(fps * 1.25))
    outro_hold_frames = int(round(fps * 1.25))
    slowdown_factor = _compute_playback_multiplier(
        sample_count=len(times),
        fps=fps,
        intro_hold_frames=intro_hold_frames,
        outro_hold_frames=outro_hold_frames,
        min_duration_s=24.0,
        min_multiplier=4,
    )

    all_xy = np.vstack(
        [
            true_positions[:, :2],
            spoofed_positions[:, :2],
            estimated_positions[:, :2],
            interceptor_positions[:, :2],
            safe_zone[:2].reshape(1, 2),
        ]
    )
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    span_xy = np.maximum(max_xy - min_xy, 1.0)

    with build_video_writers(output=output, fps=fps, frame_size=frame_size) as writers:
        intro = np.full((frame_size[1], frame_size[0], 3), (140, 74, 16), dtype=np.uint8)
        _draw_blueprint_background(intro)
        _draw_titles(intro)
        _draw_intro_card(intro)
        for _ in range(max(intro_hold_frames, 1)):
            writers.write(intro)

        for index in range(max(len(times) - 1, 1)):
            for subframe in range(slowdown_factor):
                alpha = subframe / slowdown_factor
                sample_index = min(index + 1, len(times) - 1)
                true_stack = np.vstack([true_positions[: index + 1], _interpolate_state(true_positions[index], true_positions[sample_index], alpha)])
                spoofed_stack = np.vstack([spoofed_positions[: index + 1], _interpolate_state(spoofed_positions[index], spoofed_positions[sample_index], alpha)])
                estimated_stack = np.vstack([estimated_positions[: index + 1], _interpolate_state(estimated_positions[index], estimated_positions[sample_index], alpha)])
                interceptor_stack = np.vstack([interceptor_positions[: index + 1], _interpolate_state(interceptor_positions[index], interceptor_positions[sample_index], alpha)])
                blended_time = _lerp_scalar(times[index], times[sample_index], alpha)
                blended_drift = _lerp_scalar(drift_rates[index], drift_rates[sample_index], alpha)
                blended_tracking = _lerp_scalar(tracking_errors[index], tracking_errors[sample_index], alpha)
                frame = np.full((frame_size[1], frame_size[0], 3), (140, 74, 16), dtype=np.uint8)
                _draw_blueprint_background(frame)
                _draw_titles(frame)
                _draw_stage_caption(frame, blended_time, float(times[-1]))
                _draw_flight_scene(
                    frame=frame,
                    true_positions=true_stack,
                    spoofed_positions=spoofed_stack,
                    estimated_positions=estimated_stack,
                    interceptor_positions=interceptor_stack,
                    safe_zone=safe_zone,
                    min_xy=min_xy,
                    span_xy=span_xy,
                )
                _draw_inset_chart(
                    frame=frame,
                    times=times[: sample_index + 1],
                    true_positions=true_positions[: sample_index + 1],
                    spoofed_positions=spoofed_positions[: sample_index + 1],
                )
                _draw_metrics(
                    frame=frame,
                    time_s=blended_time,
                    drift_rate_mps=blended_drift,
                    tracking_error_m=blended_tracking,
                    safe_zone_distance_m=float(np.linalg.norm(true_stack[-1] - safe_zone)),
                    interceptor_distance_m=float(np.linalg.norm(interceptor_stack[-1] - true_stack[-1])),
                )
                writers.write(frame)
        outro = frame.copy() if len(times) > 0 else intro.copy()
        _draw_outcome_card(outro, safe_zone_distance_m=float(np.linalg.norm(true_positions[-1] - safe_zone)) if len(true_positions) > 0 else 0.0)
        for _ in range(max(outro_hold_frames, 1)):
            writers.write(outro)
    return output


def render_day9_keyframe(
    true_positions: np.ndarray,
    spoofed_positions: np.ndarray,
    estimated_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    safe_zone: np.ndarray,
    output_path: str | Path,
    frame_size: tuple[int, int] = (1280, 720),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_xy = np.vstack(
        [
            true_positions[:, :2],
            spoofed_positions[:, :2],
            estimated_positions[:, :2],
            interceptor_positions[:, :2],
            safe_zone[:2].reshape(1, 2),
        ]
    )
    min_xy = np.min(all_xy, axis=0)
    max_xy = np.max(all_xy, axis=0)
    span_xy = np.maximum(max_xy - min_xy, 1.0)
    frame = np.full((frame_size[1], frame_size[0], 3), (140, 74, 16), dtype=np.uint8)
    _draw_blueprint_background(frame)
    _draw_titles(frame)
    _draw_flight_scene(
        frame=frame,
        true_positions=true_positions,
        spoofed_positions=spoofed_positions,
        estimated_positions=estimated_positions,
        interceptor_positions=interceptor_positions,
        safe_zone=safe_zone,
        min_xy=min_xy,
        span_xy=span_xy,
    )
    _draw_inset_chart(
        frame=frame,
        times=np.arange(len(true_positions), dtype=float),
        true_positions=true_positions,
        spoofed_positions=spoofed_positions,
    )
    cv2.imwrite(str(output), frame)
    return output


def _draw_blueprint_background(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    for x in range(0, width, 48):
        cv2.line(frame, (x, 0), (x, height), (170, 96, 28), 1)
    for y in range(0, height, 48):
        cv2.line(frame, (0, y), (width, y), (170, 96, 28), 1)
    cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (235, 235, 230), 2)


def _draw_titles(frame: np.ndarray) -> None:
    cv2.putText(frame, "DRONE A", (54, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.putText(frame, "INTERCEPTOR", (54, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "DRONE B", (54, 492), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 245, 245), 3, cv2.LINE_AA)
    cv2.putText(frame, "TARGET", (54, 534), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "DP5 COORDINATE DRIFT REPLAY", (820, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (245, 245, 245), 2, cv2.LINE_AA)


def _draw_flight_scene(
    frame: np.ndarray,
    true_positions: np.ndarray,
    spoofed_positions: np.ndarray,
    estimated_positions: np.ndarray,
    interceptor_positions: np.ndarray,
    safe_zone: np.ndarray,
    min_xy: np.ndarray,
    span_xy: np.ndarray,
) -> None:
    panel = (40, 140, 845, 660)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (225, 225, 225), 1)

    if len(interceptor_positions) > 1:
        _draw_curve(frame, interceptor_positions[:, :2], panel, min_xy, span_xy, (235, 235, 235), 2)
    if len(true_positions) > 1:
        _draw_curve(frame, true_positions[:, :2], panel, min_xy, span_xy, (255, 188, 94), 2)
    if len(spoofed_positions) > 1:
        _draw_curve(frame, spoofed_positions[:, :2], panel, min_xy, span_xy, (89, 189, 255), 2)
    if len(estimated_positions) > 1:
        _draw_curve(frame, estimated_positions[:, :2], panel, min_xy, span_xy, (126, 240, 160), 2)

    safe_center = _map_point(safe_zone[:2], panel, min_xy, span_xy)
    cv2.circle(frame, safe_center, 72, (240, 240, 240), 2)
    cv2.putText(frame, "SAFE", (safe_center[0] - 42, safe_center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "AREA", (safe_center[0] - 40, safe_center[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2, cv2.LINE_AA)

    if len(interceptor_positions) > 0:
        interceptor_point = _map_point(interceptor_positions[-1, :2], panel, min_xy, span_xy)
        _draw_drone_icon(frame, interceptor_point, scale=1.0, color=(245, 245, 245))
    if len(true_positions) > 0:
        target_point = _map_point(true_positions[-1, :2], panel, min_xy, span_xy)
        _draw_drone_icon(frame, target_point, scale=1.0, color=(220, 220, 220))
        cv2.circle(frame, target_point, 5, (84, 84, 255), -1)
    if len(spoofed_positions) > 0:
        spoof_point = _map_point(spoofed_positions[-1, :2], panel, min_xy, span_xy)
        cv2.circle(frame, spoof_point, 4, (89, 189, 255), -1)
    if len(estimated_positions) > 0:
        estimate_point = _map_point(estimated_positions[-1, :2], panel, min_xy, span_xy)
        cv2.circle(frame, estimate_point, 4, (126, 240, 160), -1)
    if len(true_positions) > 0:
        _draw_direction_arrow(frame, _map_point(true_positions[-1, :2], panel, min_xy, span_xy), safe_center)


def _draw_inset_chart(
    frame: np.ndarray,
    times: np.ndarray,
    true_positions: np.ndarray,
    spoofed_positions: np.ndarray,
) -> None:
    panel = (895, 410, 1230, 670)
    cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (235, 235, 235), 2)
    cv2.putText(frame, "ACTUAL POSITION", (panel[0] + 34, panel[1] + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "SPOOFED POSITION", (panel[0] + 34, panel[3] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "TIME (SECONDS)", (panel[0] + 96, panel[3] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(frame, (panel[0] + 42, panel[1] + 185), (panel[2] - 22, panel[1] + 185), (230, 230, 230), 2)
    cv2.line(frame, (panel[0] + 42, panel[3] - 42), (panel[0] + 42, panel[1] + 30), (230, 230, 230), 2)

    if len(times) < 2:
        return
    actual = np.linalg.norm(true_positions[:, :2] - true_positions[:1, :2], axis=1)
    spoofed = np.linalg.norm(spoofed_positions[:, :2] - true_positions[:1, :2], axis=1)
    _draw_series(frame, times, actual, panel, (255, 188, 94))
    _draw_series(frame, times, spoofed, panel, (89, 189, 255))


def _draw_metrics(
    frame: np.ndarray,
    time_s: float,
    drift_rate_mps: float,
    tracking_error_m: float,
    safe_zone_distance_m: float,
    interceptor_distance_m: float,
) -> None:
    lines = [
        f"t = {time_s:5.1f}s",
        f"spoofing gradient = {drift_rate_mps:0.3f} m/s",
        f"tracking error = {tracking_error_m:0.3f} m",
        f"target to safe area = {safe_zone_distance_m:0.2f} m",
        f"interceptor to target = {interceptor_distance_m:0.2f} m",
        "simulation-only DP5 redirection validation",
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (900, 150 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)


def _draw_intro_card(frame: np.ndarray) -> None:
    lines = [
        "Mission Replay Layers",
        "White: interceptor path",
        "Orange: true target trajectory",
        "Blue: spoofed navigation solution",
        "Green: tracked / fused estimate",
        "Inset: actual vs spoofed displacement over time",
    ]
    cv2.rectangle(frame, (840, 120), (1235, 340), (228, 208, 168), 1)
    for index, line in enumerate(lines):
        scale = 0.72 if index == 0 else 0.56
        weight = 2 if index == 0 else 1
        cv2.putText(frame, line, (865, 160 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), weight, cv2.LINE_AA)


def _draw_stage_caption(frame: np.ndarray, time_s: float, total_time_s: float) -> None:
    progress = 0.0 if total_time_s <= 0.0 else float(np.clip(time_s / total_time_s, 0.0, 1.0))
    if progress < 0.22:
        stage = "TARGET DETECTED"
    elif progress < 0.48:
        stage = "TRACK + FILTER"
    elif progress < 0.74:
        stage = "DRIFT ACCUMULATION"
    else:
        stage = "SAFE-AREA REDIRECTION"
    cv2.rectangle(frame, (842, 350), (1185, 388), (212, 180, 136), -1)
    cv2.putText(frame, stage, (860, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (52, 24, 10), 2, cv2.LINE_AA)


def _draw_outcome_card(frame: np.ndarray, safe_zone_distance_m: float) -> None:
    cv2.rectangle(frame, (840, 120), (1235, 340), (170, 120, 60), -1)
    cv2.rectangle(frame, (840, 120), (1235, 340), (235, 225, 210), 2)
    lines = [
        "Outcome",
        f"Final safe-area distance: {safe_zone_distance_m:0.2f} m",
        "The scene is rendered from the backend",
        "trajectory traces, not a static image.",
    ]
    for index, line in enumerate(lines):
        scale = 0.75 if index == 0 else 0.56
        weight = 2 if index == 0 else 1
        cv2.putText(frame, line, (865, 170 + index * 40), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), weight, cv2.LINE_AA)


def _draw_drone_icon(frame: np.ndarray, center: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    x, y = center
    arm = int(28 * scale)
    body = int(16 * scale)
    rotor = int(11 * scale)
    cv2.circle(frame, (x, y), body, color, 2)
    for dx, dy in ((-arm, -arm), (arm, -arm), (-arm, arm), (arm, arm)):
        cv2.line(frame, (x, y), (x + dx, y + dy), color, 2)
        cv2.circle(frame, (x + dx, y + dy), rotor, color, 2)


def _draw_direction_arrow(frame: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    cv2.arrowedLine(frame, start, end, (240, 240, 240), 2, cv2.LINE_AA, tipLength=0.03)


def _draw_curve(
    frame: np.ndarray,
    positions_xy: np.ndarray,
    panel: tuple[int, int, int, int],
    min_xy: np.ndarray,
    span_xy: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(positions_xy) < 2:
        return
    points = np.asarray([_map_point(point, panel, min_xy, span_xy) for point in positions_xy], dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)


def _draw_series(
    frame: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    panel: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = panel
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    value_span = max(max_value - min_value, 1e-6)
    time_span = max(float(times[-1] - times[0]), 1e-6)
    points: list[tuple[int, int]] = []
    for time_s, value in zip(times, values, strict=False):
        x = int((x0 + 44) + ((time_s - times[0]) / time_span) * max((x1 - x0) - 72, 1))
        y = int((y1 - 48) - ((value - min_value) / value_span) * max((y1 - y0) - 96, 1))
        points.append((x, y))
    cv2.polylines(frame, [np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))], False, color, 2, cv2.LINE_AA)


def _map_point(
    point_xy: np.ndarray,
    panel: tuple[int, int, int, int],
    min_xy: np.ndarray,
    span_xy: np.ndarray,
) -> tuple[int, int]:
    x0, y0, x1, y1 = panel
    x_norm = (float(point_xy[0]) - float(min_xy[0])) / float(span_xy[0])
    y_norm = (float(point_xy[1]) - float(min_xy[1])) / float(span_xy[1])
    x = int(x0 + 35 + x_norm * max((x1 - x0) - 70, 1))
    y = int(y1 - 35 - y_norm * max((y1 - y0) - 70, 1))
    return x, y


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


__all__ = ["render_day9_demo_video", "render_day9_keyframe"]
