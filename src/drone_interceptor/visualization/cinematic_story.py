from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drone_interceptor.visualization.video import build_video_writers


@dataclass(frozen=True, slots=True)
class CinematicTargetSnapshot:
    name: str
    position: np.ndarray
    filtered_estimate: np.ndarray
    raw_measurement: np.ndarray
    velocity: np.ndarray
    threat_level: float
    innovation_m: float
    packet_dropped: bool
    spoofing_detected: bool
    jammed: bool


@dataclass(frozen=True, slots=True)
class CinematicReplaySnapshot:
    time_s: float
    interceptor_position: np.ndarray
    interceptor_velocity: np.ndarray
    active_stage: str
    active_target: str
    rmse_m: float
    detection_fps: float
    mission_progress: float
    targets: tuple[CinematicTargetSnapshot, ...]


class CinematicSequenceRenderer:
    """Render a replay-driven cinematic chase sequence from backend mission data."""

    def __init__(
        self,
        frame_size: tuple[int, int] = (1920, 1080),
        fps: float = 24.0,
    ) -> None:
        self._width = int(frame_size[0])
        self._height = int(frame_size[1])
        self._fps = float(fps)
        self._target_trails: dict[str, list[np.ndarray]] = {}
        self._interceptor_trail: list[np.ndarray] = []
        self._camera_anchor_xy: np.ndarray | None = None
        self._camera_heading_xy: np.ndarray | None = None
        self._history_rmse: list[float] = []
        self._history_distance: list[float] = []
        self._history_progress: list[float] = []

    def render_replay(
        self,
        replay: Any,
        output_path: str | Path,
        target_duration_s: float = 12.0,
        intro_hold_s: float = 0.9,
        outro_hold_s: float = 1.1,
        playback_speed: float = 0.5,
    ) -> Path:
        snapshots = self._build_snapshots(
            replay=replay,
            target_duration_s=target_duration_s,
            intro_hold_s=intro_hold_s,
            outro_hold_s=outro_hold_s,
            playback_speed=playback_speed,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with build_video_writers(output=output, fps=self._fps, frame_size=(self._width, self._height)) as writers:
            for snapshot in snapshots:
                writers.write(self.render_snapshot(snapshot, replay_validation=getattr(replay, "validation", {})))

        return output

    def render_snapshot(
        self,
        snapshot: CinematicReplaySnapshot,
        replay_validation: dict[str, Any] | None = None,
    ) -> np.ndarray:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._draw_background(frame, snapshot)
        self._update_camera(snapshot)
        self._draw_world(frame, snapshot)
        self._draw_status_hud(frame, snapshot, replay_validation or {})
        self._draw_performance_panel(frame, snapshot)
        self._draw_vignette(frame)
        return frame

    def _build_snapshots(
        self,
        replay: Any,
        target_duration_s: float,
        intro_hold_s: float,
        outro_hold_s: float,
        playback_speed: float,
    ) -> list[CinematicReplaySnapshot]:
        frames = list(getattr(replay, "frames", ()))
        if not frames:
            return [
                CinematicReplaySnapshot(
                    time_s=0.0,
                    interceptor_position=np.array([0.0, 0.0, 110.0], dtype=float),
                    interceptor_velocity=np.zeros(3, dtype=float),
                    active_stage="Detection",
                    active_target="n/a",
                    rmse_m=0.0,
                    detection_fps=0.0,
                    mission_progress=0.0,
                    targets=tuple(),
                )
            ]

        times = np.asarray([float(frame.time_s) for frame in frames], dtype=float)
        if len(times) == 1:
            mission_duration = max(float(target_duration_s) - intro_hold_s - outro_hold_s, 0.0) * float(playback_speed)
        else:
            mission_duration = max(float(times[-1] - times[0]), 1e-6)

        render_duration = max(float(target_duration_s), intro_hold_s + outro_hold_s + (mission_duration / max(playback_speed, 1e-3)))
        frame_count = max(int(round(render_duration * self._fps)), len(frames) * 4)
        intro_frames = max(int(round(intro_hold_s * self._fps)), 1)
        outro_frames = max(int(round(outro_hold_s * self._fps)), 1)
        mission_frames = max(frame_count - intro_frames - outro_frames, 1)

        snapshots: list[CinematicReplaySnapshot] = []
        first_snapshot = self._snapshot_from_frame(frames[0], mission_progress=0.0)
        last_snapshot = self._snapshot_from_frame(frames[-1], mission_progress=1.0)

        for _ in range(intro_frames):
            snapshots.append(first_snapshot)

        for index in range(mission_frames):
            if mission_frames <= 1:
                source_t = times[-1]
                progress = 1.0
            else:
                source_t = times[0] + (index / (mission_frames - 1)) * (times[-1] - times[0]) * float(playback_speed)
                source_t = min(max(source_t, times[0]), times[-1])
                progress = index / max(mission_frames - 1, 1)
            snapshots.append(self._interpolate_snapshot(frames, times, source_t, progress))

        for _ in range(outro_frames):
            snapshots.append(last_snapshot)

        return snapshots

    def _snapshot_from_frame(self, frame: Any, mission_progress: float) -> CinematicReplaySnapshot:
        targets = tuple(
            CinematicTargetSnapshot(
                name=str(target.name),
                position=np.asarray(target.position, dtype=float).copy(),
                filtered_estimate=np.asarray(target.filtered_estimate, dtype=float).copy(),
                raw_measurement=np.asarray(target.raw_measurement, dtype=float).copy(),
                velocity=np.asarray(target.velocity, dtype=float).copy(),
                threat_level=float(target.threat_level),
                innovation_m=float(target.innovation_m),
                packet_dropped=bool(target.packet_dropped),
                spoofing_detected=bool(target.spoofing_detected),
                jammed=bool(target.jammed),
            )
            for target in frame.targets
        )
        return CinematicReplaySnapshot(
            time_s=float(frame.time_s),
            interceptor_position=np.asarray(frame.interceptor_position, dtype=float).copy(),
            interceptor_velocity=np.asarray(frame.interceptor_velocity, dtype=float).copy(),
            active_stage=str(frame.active_stage),
            active_target=str(frame.active_target),
            rmse_m=float(frame.rmse_m),
            detection_fps=float(frame.detection_fps),
            mission_progress=float(mission_progress),
            targets=targets,
        )

    def _interpolate_snapshot(
        self,
        frames: list[Any],
        times: np.ndarray,
        source_t: float,
        mission_progress: float,
    ) -> CinematicReplaySnapshot:
        if source_t <= float(times[0]):
            return self._snapshot_from_frame(frames[0], mission_progress=mission_progress)
        if source_t >= float(times[-1]):
            return self._snapshot_from_frame(frames[-1], mission_progress=mission_progress)

        upper_index = int(np.searchsorted(times, source_t, side="right"))
        lower_index = max(upper_index - 1, 0)
        frame_a = frames[lower_index]
        frame_b = frames[min(upper_index, len(frames) - 1)]
        t0 = float(times[lower_index])
        t1 = float(times[min(upper_index, len(times) - 1)])
        alpha = 0.0 if t1 <= t0 else float((source_t - t0) / max(t1 - t0, 1e-6))

        targets_by_name_a = {str(target.name): target for target in frame_a.targets}
        targets_by_name_b = {str(target.name): target for target in frame_b.targets}
        target_names = list(targets_by_name_a.keys())
        for name in targets_by_name_b:
            if name not in targets_by_name_a:
                target_names.append(name)

        targets: list[CinematicTargetSnapshot] = []
        for name in target_names:
            target_a = targets_by_name_a.get(name, next(iter(targets_by_name_a.values())))
            target_b = targets_by_name_b.get(name, target_a)
            targets.append(
                CinematicTargetSnapshot(
                    name=name,
                    position=_lerp_array(target_a.position, target_b.position, alpha),
                    filtered_estimate=_lerp_array(target_a.filtered_estimate, target_b.filtered_estimate, alpha),
                    raw_measurement=_lerp_array(target_a.raw_measurement, target_b.raw_measurement, alpha),
                    velocity=_lerp_array(target_a.velocity, target_b.velocity, alpha),
                    threat_level=_lerp_float(target_a.threat_level, target_b.threat_level, alpha),
                    innovation_m=_lerp_float(target_a.innovation_m, target_b.innovation_m, alpha),
                    packet_dropped=bool(target_a.packet_dropped if alpha < 0.5 else target_b.packet_dropped),
                    spoofing_detected=bool(target_a.spoofing_detected if alpha < 0.5 else target_b.spoofing_detected),
                    jammed=bool(target_a.jammed or target_b.jammed),
                )
            )

        return CinematicReplaySnapshot(
            time_s=_lerp_float(float(frame_a.time_s), float(frame_b.time_s), alpha),
            interceptor_position=_lerp_array(frame_a.interceptor_position, frame_b.interceptor_position, alpha),
            interceptor_velocity=_lerp_array(frame_a.interceptor_velocity, frame_b.interceptor_velocity, alpha),
            active_stage=str(frame_b.active_stage if alpha >= 0.5 else frame_a.active_stage),
            active_target=str(frame_b.active_target if alpha >= 0.5 else frame_a.active_target),
            rmse_m=_lerp_float(float(frame_a.rmse_m), float(frame_b.rmse_m), alpha),
            detection_fps=_lerp_float(float(frame_a.detection_fps), float(frame_b.detection_fps), alpha),
            mission_progress=float(mission_progress),
            targets=tuple(targets),
        )

    def _update_camera(self, snapshot: CinematicReplaySnapshot) -> None:
        interceptor_xy = np.asarray(snapshot.interceptor_position[:2], dtype=float)
        target = self._active_target(snapshot)
        if target is None:
            focus_xy = interceptor_xy.copy()
        else:
            focus_xy = np.asarray(target.filtered_estimate[:2], dtype=float)
        camera_target = 0.62 * interceptor_xy + 0.38 * focus_xy
        if self._camera_anchor_xy is None:
            self._camera_anchor_xy = camera_target.copy()
        else:
            self._camera_anchor_xy = 0.88 * self._camera_anchor_xy + 0.12 * camera_target

        velocity = np.asarray(snapshot.interceptor_velocity[:2], dtype=float)
        if float(np.linalg.norm(velocity)) < 1e-3:
            if target is not None:
                velocity = np.asarray(target.filtered_estimate[:2] - interceptor_xy, dtype=float)
        heading = _normalize2d(velocity)
        if self._camera_heading_xy is None:
            self._camera_heading_xy = heading
        else:
            self._camera_heading_xy = _normalize2d(0.85 * self._camera_heading_xy + 0.15 * heading)

    def _draw_background(self, frame: np.ndarray, snapshot: CinematicReplaySnapshot) -> None:
        height, width = frame.shape[:2]
        sky_top = np.array([16, 32, 58], dtype=float)
        sky_mid = np.array([66, 100, 152], dtype=float)
        ground_top = np.array([20, 35, 22], dtype=float)
        ground_bottom = np.array([8, 14, 14], dtype=float)
        horizon = int(height * 0.58)
        for y in range(height):
            if y < horizon:
                alpha = y / max(horizon - 1, 1)
                color = ((1.0 - alpha) * sky_top) + (alpha * sky_mid)
            else:
                alpha = (y - horizon) / max(height - horizon - 1, 1)
                color = ((1.0 - alpha) * ground_top) + (alpha * ground_bottom)
            frame[y, :, :] = np.clip(color, 0, 255).astype(np.uint8)

        self._draw_sun_and_clouds(frame, snapshot)
        self._draw_world_grid(frame, snapshot)
        self._draw_silhouette_band(frame)

    def _draw_sun_and_clouds(self, frame: np.ndarray, snapshot: CinematicReplaySnapshot) -> None:
        height, width = frame.shape[:2]
        drift = int(30.0 * np.sin(snapshot.time_s * 0.12))
        sun_center = (int(width * 0.82) + drift, int(height * 0.16))
        cv2.circle(frame, sun_center, 80, (120, 205, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, sun_center, 120, (80, 140, 210), 3, cv2.LINE_AA)
        for index in range(3):
            cx = int(width * (0.22 + 0.23 * index) + 18.0 * np.sin(snapshot.time_s * 0.05 + index))
            cy = int(height * (0.19 + 0.03 * np.cos(snapshot.time_s * 0.07 + index)))
            cv2.ellipse(frame, (cx, cy), (90, 34), 0, 0, 360, (162, 188, 230), -1, cv2.LINE_AA)
            cv2.ellipse(frame, (cx + 44, cy + 4), (70, 28), 0, 0, 360, (166, 194, 235), -1, cv2.LINE_AA)

    def _draw_world_grid(self, frame: np.ndarray, snapshot: CinematicReplaySnapshot) -> None:
        height, width = frame.shape[:2]
        horizon = int(height * 0.58)
        vanishing = (width // 2, horizon)
        camera = self._camera_anchor_xy if self._camera_anchor_xy is not None else np.array([0.0, 0.0], dtype=float)
        for offset in range(-8, 9):
            x = int(width * 0.5 + offset * width * 0.08)
            x += int(12.0 * np.sin(snapshot.time_s * 0.17 + offset * 0.3))
            cv2.line(frame, (x, height), vanishing, (54, 82, 88), 1, cv2.LINE_AA)
        for row in range(10):
            y = int(horizon + ((row + 1) / 10.0) ** 1.9 * (height - horizon - 24))
            width_half = int((y - horizon) * (1.3 + 0.02 * np.linalg.norm(camera)))
            cv2.line(frame, (vanishing[0] - width_half, y), (vanishing[0] + width_half, y), (42, 70, 72), 1, cv2.LINE_AA)

    def _draw_silhouette_band(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        horizon = int(height * 0.58)
        skyline = [
            (0.04, 0.18), (0.10, 0.12), (0.17, 0.22), (0.24, 0.14), (0.31, 0.19),
            (0.40, 0.16), (0.49, 0.24), (0.61, 0.13), (0.68, 0.20), (0.77, 0.15), (0.88, 0.22),
        ]
        for x_ratio, height_ratio in skyline:
            x = int(width * x_ratio)
            building_h = int(frame.shape[0] * height_ratio)
            building_w = int(width * 0.036)
            cv2.rectangle(frame, (x, horizon - building_h), (x + building_w, horizon), (17, 22, 36), -1)
            cv2.rectangle(frame, (x, horizon - building_h), (x + building_w, horizon), (34, 46, 66), 1)

    def _draw_world(self, frame: np.ndarray, snapshot: CinematicReplaySnapshot) -> None:
        height, width = frame.shape[:2]
        camera_xy = self._camera_anchor_xy if self._camera_anchor_xy is not None else np.asarray(snapshot.interceptor_position[:2], dtype=float)
        heading_xy = self._camera_heading_xy if self._camera_heading_xy is not None else np.array([1.0, 0.0], dtype=float)
        heading_xy = _normalize2d(heading_xy)
        right_xy = np.array([heading_xy[1], -heading_xy[0]], dtype=float)
        camera_height = float(snapshot.interceptor_position[2]) + 36.0
        focal = 660.0
        depth_bias = 210.0
        horizon = int(height * 0.59)

        self._interceptor_trail.append(np.asarray(snapshot.interceptor_position, dtype=float).copy())
        self._interceptor_trail = self._interceptor_trail[-28:]

        for target in snapshot.targets:
            trail = self._target_trails.setdefault(target.name, [])
            trail.append(np.asarray(target.position, dtype=float).copy())
            self._target_trails[target.name] = trail[-28:]

        ordered_targets = sorted(snapshot.targets, key=lambda item: (item.jammed, -item.threat_level))
        for target in ordered_targets:
            projected = self._project_point(
                point=np.asarray(target.position, dtype=float),
                camera_xy=camera_xy,
                right_xy=right_xy,
                heading_xy=heading_xy,
                camera_height=camera_height,
                focal=focal,
                depth_bias=depth_bias,
                horizon=horizon,
            )
            if projected is None:
                continue
            center, scale, depth = projected
            trail_points = []
            for point in self._target_trails.get(target.name, []):
                trail_projection = self._project_point(
                    point=point,
                    camera_xy=camera_xy,
                    right_xy=right_xy,
                    heading_xy=heading_xy,
                    camera_height=camera_height,
                    focal=focal,
                    depth_bias=depth_bias,
                    horizon=horizon,
                )
                if trail_projection is not None:
                    trail_points.append(trail_projection[0])
            if len(trail_points) > 1:
                cv2.polylines(frame, [np.asarray(trail_points, dtype=np.int32).reshape((-1, 1, 2))], False, _target_color(target, active=(target.name == snapshot.active_target)), 2, cv2.LINE_AA)
            self._draw_target_vehicle(frame, center=center, scale=scale, target=target, active=(target.name == snapshot.active_target))
            self._draw_target_metadata(frame, center=center, target=target, depth=depth)
            if target.name == snapshot.active_target:
                self._draw_target_reticle(frame, center=center, scale=scale)
                lead = np.asarray(target.filtered_estimate[:2], dtype=float) + np.asarray(target.velocity[:2], dtype=float) * 2.4
                lead_proj = self._project_point(
                    point=np.array([lead[0], lead[1], target.position[2]], dtype=float),
                    camera_xy=camera_xy,
                    right_xy=right_xy,
                    heading_xy=heading_xy,
                    camera_height=camera_height,
                    focal=focal,
                    depth_bias=depth_bias,
                    horizon=horizon,
                )
                if lead_proj is not None:
                    cv2.line(frame, center, lead_proj[0], (255, 214, 88), 2, cv2.LINE_AA)
                    cv2.circle(frame, lead_proj[0], max(int(18 * scale), 8), (255, 214, 88), 2, cv2.LINE_AA)

        interceptor_proj = self._project_point(
            point=np.asarray(snapshot.interceptor_position, dtype=float),
            camera_xy=camera_xy,
            right_xy=right_xy,
            heading_xy=heading_xy,
            camera_height=camera_height,
            focal=focal,
            depth_bias=depth_bias,
            horizon=horizon,
        )
        if interceptor_proj is not None:
            self._draw_interceptor_vehicle(frame, center=interceptor_proj[0], scale=interceptor_proj[1], heading_xy=heading_xy)
            cv2.line(frame, interceptor_proj[0], (width // 2, int(height * 0.68)), (108, 210, 255), 1, cv2.LINE_AA)

    def _project_point(
        self,
        point: np.ndarray,
        camera_xy: np.ndarray,
        right_xy: np.ndarray,
        heading_xy: np.ndarray,
        camera_height: float,
        focal: float,
        depth_bias: float,
        horizon: int,
    ) -> tuple[tuple[int, int], float, float] | None:
        rel_xy = np.asarray(point[:2], dtype=float) - np.asarray(camera_xy, dtype=float)
        depth = float(np.dot(rel_xy, heading_xy))
        lateral = float(np.dot(rel_xy, right_xy))
        if depth < -40.0:
            return None
        perspective = focal / max(depth + depth_bias, 80.0)
        scale = float(np.clip(perspective * 1.35, 0.32, 1.85))
        x = int(self._width * 0.5 + lateral * perspective)
        y = int(horizon - (float(point[2]) - camera_height) * perspective * 2.4 - depth * 0.22)
        if x < -160 or x > self._width + 160 or y < -160 or y > self._height + 160:
            return None
        return (x, y), scale, depth

    def _draw_interceptor_vehicle(self, frame: np.ndarray, center: tuple[int, int], scale: float, heading_xy: np.ndarray) -> None:
        x, y = center
        heading_angle = float(np.degrees(np.arctan2(-heading_xy[1], heading_xy[0])))
        body_color = (238, 246, 252)
        glow_color = (90, 210, 255)
        arm = max(int(70 * scale), 24)
        rotor = max(int(30 * scale), 12)
        cv2.ellipse(frame, (x, y), (max(int(68 * scale), 24), max(int(24 * scale), 10)), heading_angle, 0, 360, body_color, 3, cv2.LINE_AA)
        cv2.line(frame, (x - int(58 * scale), y), (x + int(58 * scale), y), body_color, 3, cv2.LINE_AA)
        cv2.line(frame, (x, y - int(20 * scale)), (x, y + int(20 * scale)), body_color, 3, cv2.LINE_AA)
        for dx, dy in ((-arm, -arm // 2), (arm, -arm // 2), (-arm // 2, arm // 2), (arm // 2, arm // 2)):
            rotor_center = (x + dx, y + dy)
            cv2.line(frame, (x, y), rotor_center, body_color, 2, cv2.LINE_AA)
            cv2.ellipse(frame, rotor_center, (rotor, max(int(12 * scale), 5)), heading_angle, 0, 360, glow_color, 2, cv2.LINE_AA)
            cv2.circle(frame, rotor_center, max(int(5 * scale), 2), body_color, -1, cv2.LINE_AA)

    def _draw_target_vehicle(self, frame: np.ndarray, center: tuple[int, int], scale: float, target: CinematicTargetSnapshot, active: bool) -> None:
        x, y = center
        color = _target_color(target, active=active)
        body = max(int(18 * scale), 7)
        rotor = max(int(8 * scale), 4)
        arm = max(int(30 * scale), 12)
        cv2.ellipse(frame, (x, y), (body, max(int(10 * scale), 4)), 0, 0, 360, color, 2, cv2.LINE_AA)
        for dx, dy in ((-arm, -arm // 2), (arm, -arm // 2), (-arm // 2, arm // 2), (arm // 2, arm // 2)):
            rotor_center = (x + dx, y + dy)
            cv2.line(frame, (x, y), rotor_center, color, 2, cv2.LINE_AA)
            cv2.ellipse(frame, rotor_center, (rotor, max(int(5 * scale), 2)), 0, 0, 360, color, 2, cv2.LINE_AA)
        if target.packet_dropped:
            cv2.circle(frame, (x, y), max(int(12 * scale), 6), (255, 85, 85), 2, cv2.LINE_AA)
        if target.spoofing_detected:
            cv2.circle(frame, (x, y), max(int(22 * scale), 10), (255, 214, 88), 2, cv2.LINE_AA)

    def _draw_target_metadata(self, frame: np.ndarray, center: tuple[int, int], target: CinematicTargetSnapshot, depth: float) -> None:
        x, y = center
        label_color = _target_color(target, active=(target.name == self._current_active_target()))
        threat_text = f"{target.name}  thr {target.threat_level:0.3f}"
        innovation_text = f"innovation {target.innovation_m:0.2f}m"
        depth_text = f"depth {depth:0.1f}m"
        cv2.putText(frame, threat_text, (x - 42, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, label_color, 2, cv2.LINE_AA)
        cv2.putText(frame, innovation_text, (x - 52, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (245, 248, 250), 1, cv2.LINE_AA)
        cv2.putText(frame, depth_text, (x - 42, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (185, 215, 235), 1, cv2.LINE_AA)

    def _draw_target_reticle(self, frame: np.ndarray, center: tuple[int, int], scale: float) -> None:
        x, y = center
        radius = max(int(38 * scale), 16)
        cv2.circle(frame, (x, y), radius, (255, 214, 88), 2, cv2.LINE_AA)
        cv2.circle(frame, (x, y), max(radius - 8, 10), (255, 214, 88), 1, cv2.LINE_AA)
        cv2.line(frame, (x - radius - 8, y), (x - radius + 4, y), (255, 214, 88), 2, cv2.LINE_AA)
        cv2.line(frame, (x + radius - 4, y), (x + radius + 8, y), (255, 214, 88), 2, cv2.LINE_AA)
        cv2.line(frame, (x, y - radius - 8), (x, y - radius + 4), (255, 214, 88), 2, cv2.LINE_AA)
        cv2.line(frame, (x, y + radius - 4), (x, y + radius + 8), (255, 214, 88), 2, cv2.LINE_AA)

    def _draw_status_hud(self, frame: np.ndarray, snapshot: CinematicReplaySnapshot, validation: dict[str, Any]) -> None:
        cv2.putText(frame, "DRONE INTERCEPTOR / CHASE CAM", (42, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (146, 255, 195), 2, cv2.LINE_AA)
        cv2.putText(frame, "simulation-driven replay", (44, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (205, 237, 255), 2, cv2.LINE_AA)
        status_lines = [
            f"stage: {snapshot.active_stage}",
            f"target: {snapshot.active_target}",
            f"time: {snapshot.time_s:05.1f}s",
            f"rmse: {snapshot.rmse_m:0.3f} m",
            f"detection fps: {snapshot.detection_fps:0.1f}",
        ]
        if validation:
            if validation.get("success") is True:
                status_lines.append("mission: intercept achieved")
            elif validation.get("success") is False:
                status_lines.append("mission: interception active")
            if validation.get("packet_loss_events") is not None:
                status_lines.append(f"packet loss: {int(validation.get('packet_loss_events', 0))}")
        for index, line in enumerate(status_lines):
            cv2.putText(frame, line, (44, 146 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 248, 250), 2, cv2.LINE_AA)

    def _draw_performance_panel(self, frame: np.ndarray, snapshot: CinematicReplaySnapshot) -> None:
        height, width = frame.shape[:2]
        panel = (width - 470, height - 270, width - 36, height - 38)
        cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(frame, "MOTION TRACK", (panel[0] + 24, panel[1] + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
        if len(self._history_rmse) > 1:
            self._draw_series(frame, panel, self._history_rmse, color=(255, 188, 94), label="rmse")
        if len(self._history_distance) > 1:
            self._draw_series(frame, panel, self._history_distance, color=(90, 189, 255), label="distance")
        current_distance = self._current_distance(snapshot)
        self._history_rmse.append(float(snapshot.rmse_m))
        self._history_distance.append(current_distance)
        self._history_progress.append(float(snapshot.mission_progress))
        self._history_rmse = self._history_rmse[-180:]
        self._history_distance = self._history_distance[-180:]
        self._history_progress = self._history_progress[-180:]

    def _draw_series(self, frame: np.ndarray, panel: tuple[int, int, int, int], values: list[float], color: tuple[int, int, int], label: str) -> None:
        x0, y0, x1, y1 = panel
        pad_x = 28
        pad_y = 52
        inner_x0 = x0 + pad_x
        inner_y0 = y0 + pad_y
        inner_x1 = x1 - 18
        inner_y1 = y1 - 18
        values_np = np.asarray(values, dtype=float)
        min_v = float(np.min(values_np))
        max_v = float(np.max(values_np))
        span_v = max(max_v - min_v, 1e-6)
        idx = np.arange(len(values_np), dtype=float)
        points = []
        for i, value in zip(idx, values_np, strict=False):
            x = int(inner_x0 + (i / max(len(values_np) - 1, 1)) * (inner_x1 - inner_x0))
            y = int(inner_y1 - ((value - min_v) / span_v) * (inner_y1 - inner_y0))
            points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(frame, [np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))], False, color, 2, cv2.LINE_AA)
            cv2.circle(frame, points[-1], 4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{label}: {values_np[-1]:0.2f}", (x0 + 24, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)

    def _draw_vignette(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self._width, self._height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.06, frame, 0.94, 0.0, frame)
        cv2.rectangle(frame, (20, 20), (self._width - 20, self._height - 20), (84, 255, 176), 1, cv2.LINE_AA)

    def _active_target(self, snapshot: CinematicReplaySnapshot) -> CinematicTargetSnapshot | None:
        for target in snapshot.targets:
            if target.name == snapshot.active_target:
                return target
        return snapshot.targets[0] if snapshot.targets else None

    def _current_active_target(self) -> str:
        return "" if not self._target_trails else next(iter(self._target_trails))

    def _current_distance(self, snapshot: CinematicReplaySnapshot) -> float:
        target = self._active_target(snapshot)
        if target is None:
            return 0.0
        return float(np.linalg.norm(np.asarray(target.position, dtype=float) - np.asarray(snapshot.interceptor_position, dtype=float)))


def render_cinematic_replay(
    replay: Any,
    output_path: str | Path,
    frame_size: tuple[int, int] = (1920, 1080),
    fps: float = 24.0,
    target_duration_s: float = 12.0,
    intro_hold_s: float = 0.9,
    outro_hold_s: float = 1.1,
    playback_speed: float = 0.5,
) -> Path:
    renderer = CinematicSequenceRenderer(frame_size=frame_size, fps=fps)
    return renderer.render_replay(
        replay=replay,
        output_path=output_path,
        target_duration_s=target_duration_s,
        intro_hold_s=intro_hold_s,
        outro_hold_s=outro_hold_s,
        playback_speed=playback_speed,
    )


def _lerp_array(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return (1.0 - alpha) * left + alpha * right


def _lerp_float(left: float, right: float, alpha: float) -> float:
    return float((1.0 - alpha) * float(left) + alpha * float(right))


def _normalize2d(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float).reshape(-1)
    if vec.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=float)
    planar = vec[:2]
    norm = float(np.linalg.norm(planar))
    if norm < 1e-6:
        return np.array([1.0, 0.0], dtype=float)
    return planar / norm


def _target_color(target: CinematicTargetSnapshot, active: bool = False) -> tuple[int, int, int]:
    if target.jammed:
        return (86, 90, 255)
    if active:
        return (255, 214, 88)
    if target.spoofing_detected:
        return (255, 166, 88)
    if target.packet_dropped:
        return (120, 245, 160)
    return (84, 255, 176)


__all__ = [
    "CinematicReplaySnapshot",
    "CinematicSequenceRenderer",
    "CinematicTargetSnapshot",
    "render_cinematic_replay",
]
