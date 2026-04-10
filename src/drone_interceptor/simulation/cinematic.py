from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from drone_interceptor.visualization.sensor_gallery import draw_real_sensor_panel
from drone_interceptor.visualization.video import build_video_writers


@dataclass(slots=True)
class CinematicFrame:
    image: np.ndarray
    timestamp: str


class CinematicRecorder:
    """Capture high-grade mission video frames from AirSim or a synthetic fallback."""

    def __init__(
        self,
        client: Any | None,
        output_dir: str | Path,
        camera_name: str = "cinematic_cam",
        resolution: tuple[int, int] = (1920, 1080),
        fps: float = 60.0,
    ) -> None:
        self._client = client
        self._camera_name = str(camera_name)
        self._width = int(resolution[0])
        self._height = int(resolution[1])
        self._fps = float(fps)
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._target_history: dict[str, list[np.ndarray]] = {}
        self._last_jammed_states: dict[str, bool] = {}

    def reset_history(self) -> None:
        self._target_history.clear()
        self._last_jammed_states.clear()

    def capture_frame(
        self,
        targets: list[dict[str, Any]] | None = None,
        interceptor: np.ndarray | None = None,
        mission_time_s: float | None = None,
        active_stage: str | None = None,
        active_target: str | None = None,
        mission_metrics: dict[str, Any] | None = None,
    ) -> CinematicFrame:
        frame = self._capture_airsim_frame()
        if frame is None:
            frame = self._build_fallback_frame(
                targets=targets or [],
                interceptor=interceptor,
                mission_time_s=mission_time_s,
                active_stage=active_stage,
                active_target=active_target,
                mission_metrics=mission_metrics,
            )
        return CinematicFrame(image=frame, timestamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f"))

    def save_video(self, frames: list[np.ndarray], prefix: str = "airsim_bms_demo") -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output = self._output_dir / f"{prefix}_{timestamp}.mp4"
        if not frames:
            blank = self._build_fallback_frame(targets=[], interceptor=None)
            frames = [blank]
        output_fps = min(self._fps, 24.0)
        frames = self._expand_frames(frames=frames, fps=output_fps)
        height, width = frames[0].shape[:2]
        with build_video_writers(output=output, fps=output_fps, frame_size=(width, height)) as writers:
            for frame in frames:
                writers.write(np.asarray(frame, dtype=np.uint8))
        return output

    def _expand_frames(self, frames: list[np.ndarray], fps: float) -> list[np.ndarray]:
        if not frames:
            return [self._build_fallback_frame(targets=[], interceptor=None)]
        intro_hold_frames = int(round(fps * 1.2))
        outro_hold_frames = int(round(fps * 1.0))
        multiplier = self._compute_playback_multiplier(
            sample_count=len(frames),
            fps=fps,
            intro_hold_frames=intro_hold_frames,
            outro_hold_frames=outro_hold_frames,
            min_duration_s=22.0,
            min_multiplier=6,
        )
        expanded: list[np.ndarray] = []
        intro = frames[0].copy()
        self._draw_video_slate(
            intro,
            title="DP5 CHASE CAMERA REPLAY",
            subtitle="Backend-driven mission states rendered as cinematic fallback footage",
        )
        expanded.extend([intro.copy() for _ in range(max(intro_hold_frames, 1))])
        if len(frames) == 1:
            expanded.extend([frames[0].copy() for _ in range(max(outro_hold_frames, 1))])
            return expanded
        for index in range(len(frames) - 1):
            current = np.asarray(frames[index], dtype=np.uint8)
            nxt = np.asarray(frames[index + 1], dtype=np.uint8)
            expanded.append(current.copy())
            for subframe in range(1, multiplier):
                alpha = subframe / multiplier
                blended = cv2.addWeighted(current, 1.0 - alpha, nxt, alpha, 0.0)
                expanded.append(blended)
        outro = frames[-1].copy()
        self._draw_video_slate(
            outro,
            title="MISSION SEGMENT COMPLETE",
            subtitle="Target tracks, interceptor pose, and mission stage are sourced from the backend replay",
        )
        expanded.extend([outro.copy() for _ in range(max(outro_hold_frames, 1))])
        return expanded

    def _compute_playback_multiplier(
        self,
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

    def _draw_video_slate(self, frame: np.ndarray, title: str, subtitle: str) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (42, 42), (frame.shape[1] - 42, 196), (10, 18, 26), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
        cv2.rectangle(frame, (42, 42), (frame.shape[1] - 42, 196), (90, 216, 194), 2, cv2.LINE_AA)
        cv2.putText(frame, title, (72, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.18, (240, 248, 250), 3, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (74, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (205, 236, 244), 2, cv2.LINE_AA)

    def _capture_airsim_frame(self) -> np.ndarray | None:
        if self._client is None or not hasattr(self._client, "simGetImages"):
            return None
        try:
            airsim = __import__("airsim")
            responses = self._client.simGetImages(
                [airsim.ImageRequest(self._camera_name, airsim.ImageType.SceneColor, False, False)]
            )
            if not responses:
                return None
            response = responses[0]
            if int(getattr(response, "width", 0)) <= 0 or int(getattr(response, "height", 0)) <= 0:
                return None
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(int(response.height), int(response.width), 3)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _build_fallback_frame(
        self,
        targets: list[dict[str, Any]],
        interceptor: np.ndarray | None,
        mission_time_s: float | None,
        active_stage: str | None,
        active_target: str | None,
        mission_metrics: dict[str, Any] | None,
    ) -> np.ndarray:
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        display_targets = self._prepare_display_targets(targets)
        self._draw_background(frame)
        self._draw_hud(
            frame,
            targets=display_targets,
            interceptor=interceptor,
            mission_time_s=mission_time_s,
            active_stage=active_stage,
            mission_metrics=mission_metrics,
        )
        draw_real_sensor_panel(
            frame,
            panel=(38, 246, 478, 412),
            step_index=int(round(float(mission_time_s or 0.0) * 2.0)),
            title="Real Sensor Frames",
            subtitle="VisDrone reference imagery",
        )
        self._draw_world_scene(frame, targets=display_targets, interceptor=interceptor, active_target=active_target)
        self._draw_logic_panel(
            frame,
            targets=display_targets,
            active_stage=active_stage,
            active_target=active_target,
            mission_metrics=mission_metrics,
        )
        self._draw_tactical_inset(
            frame,
            targets=display_targets,
            interceptor=np.asarray(interceptor if interceptor is not None else np.array([0.0, 0.0, 110.0], dtype=float), dtype=float),
            active_target=active_target,
        )
        self._draw_vignette(frame)
        return frame

    def _prepare_display_targets(self, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for target in targets:
            name = str(target.get("name", "Target"))
            jammed = bool(target.get("jammed", False))
            jam_transition = jammed and not self._last_jammed_states.get(name, False)
            cloned = dict(target)
            cloned["jam_transition"] = jam_transition
            prepared.append(cloned)
            self._last_jammed_states[name] = jammed
        return prepared

    def _draw_background(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        top = np.array([26, 44, 78], dtype=float)
        horizon = np.array([74, 112, 164], dtype=float)
        ground = np.array([24, 38, 32], dtype=float)
        split = int(height * 0.58)
        for y in range(height):
            if y < split:
                alpha = y / max(split - 1, 1)
                color = ((1.0 - alpha) * top) + (alpha * horizon)
            else:
                alpha = (y - split) / max((height - split) - 1, 1)
                color = ((1.0 - alpha) * ground) + (alpha * np.array([10, 16, 14], dtype=float))
            frame[y, :, :] = np.clip(color, 0, 255).astype(np.uint8)
        cv2.circle(frame, (int(width * 0.78), int(height * 0.18)), 95, (118, 198, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(width * 0.78), int(height * 0.18)), 125, (80, 120, 175), 3, cv2.LINE_AA)
        self._draw_perspective_grid(frame)
        self._draw_skyline(frame)

    def _draw_perspective_grid(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        horizon_y = int(height * 0.56)
        vanishing = (width // 2, horizon_y)
        for offset in range(-7, 8):
            bottom_x = int((width / 2) + offset * width * 0.09)
            cv2.line(frame, (bottom_x, height), vanishing, (56, 92, 88), 1, cv2.LINE_AA)
        for row in range(9):
            y = int(horizon_y + ((row + 1) / 9.0) ** 1.75 * (height - horizon_y - 24))
            width_half = int((y - horizon_y) * 1.65)
            cv2.line(frame, (vanishing[0] - width_half, y), (vanishing[0] + width_half, y), (52, 86, 82), 1, cv2.LINE_AA)

    def _draw_skyline(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        baseline = int(height * 0.56)
        skyline = [
            (0.05, 0.20), (0.12, 0.13), (0.18, 0.18), (0.26, 0.11), (0.33, 0.17),
            (0.42, 0.12), (0.51, 0.19), (0.61, 0.10), (0.68, 0.16), (0.77, 0.12), (0.88, 0.18),
        ]
        for x_ratio, h_ratio in skyline:
            x = int(width * x_ratio)
            w = int(width * 0.035)
            h = int(height * h_ratio)
            cv2.rectangle(frame, (x, baseline - h), (x + w, baseline), (20, 28, 46), -1)
            cv2.rectangle(frame, (x, baseline - h), (x + w, baseline), (36, 50, 74), 1)

    def _draw_hud(
        self,
        frame: np.ndarray,
        targets: list[dict[str, Any]],
        interceptor: np.ndarray | None,
        mission_time_s: float | None,
        active_stage: str | None,
        mission_metrics: dict[str, Any] | None,
    ) -> None:
        cv2.putText(frame, "ADVANCED MISSION FEED", (42, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 202), 2, cv2.LINE_AA)
        cv2.putText(frame, "BACKEND-DRIVEN CHASE CAMERA", (44, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (197, 236, 255), 2, cv2.LINE_AA)
        info_lines = [
            f"targets: {len(targets)}",
            f"time: {0.0 if mission_time_s is None else float(mission_time_s):05.1f}s",
            f"stage: {active_stage or 'tracking'}",
        ]
        if interceptor is not None:
            info_lines.append(
                f"interceptor xyz=({float(interceptor[0]):.1f}, {float(interceptor[1]):.1f}, {float(interceptor[2]):.1f})"
            )
        if mission_metrics is not None:
            info_lines.append(f"rmse: {float(mission_metrics.get('rmse_m', 0.0)):0.2f} m")
            info_lines.append(f"detection fps: {float(mission_metrics.get('detection_fps', 0.0)):0.1f}")
        for index, line in enumerate(info_lines):
            cv2.putText(frame, line, (44, 146 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 248, 250), 2, cv2.LINE_AA)

    def _draw_world_scene(self, frame: np.ndarray, targets: list[dict[str, Any]], interceptor: np.ndarray | None, active_target: str | None) -> None:
        if interceptor is None:
            interceptor = np.array([0.0, 0.0, 110.0], dtype=float)
        interceptor = np.asarray(interceptor, dtype=float)
        self._target_history.setdefault("__interceptor__", []).append(interceptor.copy())
        self._target_history["__interceptor__"] = self._target_history["__interceptor__"][-20:]

        width = frame.shape[1]
        height = frame.shape[0]
        self._draw_interceptor_vehicle(frame, center=(width // 2, int(height * 0.79)))

        ordered_targets = sorted(targets, key=lambda item: float(item.get("threat_level", 0.0)), reverse=True)
        for index, target in enumerate(ordered_targets):
            position = np.asarray(target.get("position", [230.0 + 45.0 * index, (index - 1.5) * 30.0, 120.0]), dtype=float)
            self._target_history.setdefault(str(target.get("name", f"Target_{index + 1}")), []).append(position.copy())
            self._target_history[str(target.get("name", f"Target_{index + 1}"))] = self._target_history[str(target.get("name", f"Target_{index + 1}"))][-18:]
            projected = self._project_relative_point(position - interceptor)
            if projected is None:
                continue
            center, scale = projected
            color = (84, 255, 176) if not bool(target.get("jammed", False)) else (76, 76, 255)
            trail = [
                self._project_relative_point(history_position - interceptor)
                for history_position in self._target_history.get(str(target.get("name", f"Target_{index + 1}")), [])
            ]
            trail_points = [item[0] for item in trail if item is not None]
            if len(trail_points) >= 2:
                cv2.polylines(frame, [np.asarray(trail_points, dtype=np.int32).reshape((-1, 1, 2))], False, color, 2, cv2.LINE_AA)
            self._draw_target_vehicle(frame, center=center, scale=scale, color=color)
            self._draw_target_label(frame, center=center, target=target, color=color)
            if bool(target.get("jammed", False)):
                self._draw_hit_marker(frame, center=center, scale=scale, fresh=bool(target.get("jam_transition", False)))
            if str(target.get("name", "")) == str(active_target):
                self._draw_target_reticle(frame, center=center, scale=scale, color=(255, 214, 92))
                cv2.line(frame, (width // 2, int(height * 0.75)), center, (255, 214, 92), 1, cv2.LINE_AA)

    def _draw_interceptor_vehicle(self, frame: np.ndarray, center: tuple[int, int]) -> None:
        x, y = center
        body_color = (232, 240, 248)
        accent = (98, 196, 255)
        rotor_offsets = [(-74, -34), (74, -34), (-46, 38), (46, 38)]
        cv2.ellipse(frame, (x, y), (68, 28), 0, 0, 360, body_color, 3, cv2.LINE_AA)
        cv2.line(frame, (x - 62, y - 8), (x + 62, y - 8), body_color, 3, cv2.LINE_AA)
        cv2.line(frame, (x - 36, y + 22), (x + 36, y + 22), body_color, 3, cv2.LINE_AA)
        for dx, dy in rotor_offsets:
            rotor_center = (x + dx, y + dy)
            cv2.line(frame, (x, y), rotor_center, body_color, 2, cv2.LINE_AA)
            cv2.ellipse(frame, rotor_center, (30, 10), 0, 0, 360, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, rotor_center, 5, body_color, -1, cv2.LINE_AA)

    def _draw_target_vehicle(self, frame: np.ndarray, center: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
        x, y = center
        arm = max(int(32 * scale), 12)
        body_w = max(int(18 * scale), 7)
        body_h = max(int(10 * scale), 5)
        rotor_r = max(int(9 * scale), 4)
        rotor_y = max(int(5 * scale), 3)
        rotor_offsets = [(-arm, -arm // 2), (arm, -arm // 2), (-arm // 2, arm // 2), (arm // 2, arm // 2)]
        cv2.ellipse(frame, (x, y), (body_w, body_h), 0, 0, 360, color, 2, cv2.LINE_AA)
        for dx, dy in rotor_offsets:
            rotor_center = (x + dx, y + dy)
            cv2.line(frame, (x, y), rotor_center, color, 2, cv2.LINE_AA)
            cv2.ellipse(frame, rotor_center, (rotor_r, rotor_y), 0, 0, 360, color, 2, cv2.LINE_AA)

    def _draw_target_label(self, frame: np.ndarray, center: tuple[int, int], target: dict[str, Any], color: tuple[int, int, int]) -> None:
        x, y = center
        name = str(target.get("name", "Target"))
        threat = float(target.get("threat_level", 0.0))
        cv2.putText(frame, name, (x - 34, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"threat {threat:0.3f}", (x - 46, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 244, 245), 1, cv2.LINE_AA)
        if bool(target.get("spoofing_detected", False)):
            cv2.putText(frame, "EKF gate", (x - 34, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 214, 92), 1, cv2.LINE_AA)
        elif bool(target.get("packet_dropped", False)):
            cv2.putText(frame, "buffer hold", (x - 38, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (168, 214, 255), 1, cv2.LINE_AA)

    def _draw_target_reticle(self, frame: np.ndarray, center: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
        x, y = center
        radius = max(int(34 * scale), 18)
        cv2.circle(frame, (x, y), radius, color, 2, cv2.LINE_AA)
        cv2.line(frame, (x - radius - 10, y), (x - radius + 2, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x + radius - 2, y), (x + radius + 10, y), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y - radius - 10), (x, y - radius + 2), color, 2, cv2.LINE_AA)
        cv2.line(frame, (x, y + radius - 2), (x, y + radius + 10), color, 2, cv2.LINE_AA)

    def _draw_hit_marker(self, frame: np.ndarray, center: tuple[int, int], scale: float, fresh: bool) -> None:
        radius = max(int(26 * scale), 16)
        colors = [(88, 214, 255), (84, 176, 255), (84, 84, 255)]
        for index, color in enumerate(colors, start=1):
            cv2.circle(frame, center, radius + index * 10, color, 2 if fresh else 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "HIT CONFIRMED" if fresh else "JAMMED",
            (center[0] - 52, center[1] + radius + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 248, 250),
            1,
            cv2.LINE_AA,
        )

    def _project_relative_point(self, relative: np.ndarray) -> tuple[tuple[int, int], float] | None:
        width = self._width
        height = self._height
        depth = float(relative[0]) + 220.0
        if depth < 30.0:
            return None
        perspective = 420.0 / depth
        x = int((width * 0.5) + float(relative[1]) * perspective * 3.4)
        y = int((height * 0.72) - float(relative[2] - 95.0) * perspective * 3.0 - min(float(relative[0]), 320.0) * 0.45)
        if x < -120 or x > width + 120 or y < -120 or y > height + 120:
            return None
        scale = float(np.clip(perspective * 1.7, 0.35, 1.35))
        return (x, y), scale

    def _draw_vignette(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (90, 216, 194), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0.0, frame)

    def _draw_logic_panel(
        self,
        frame: np.ndarray,
        targets: list[dict[str, Any]],
        active_stage: str | None,
        active_target: str | None,
        mission_metrics: dict[str, Any] | None,
    ) -> None:
        panel = (1240, 120, 1860, 418)
        cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (10, 18, 28), -1)
        cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (90, 216, 194), 1, cv2.LINE_AA)
        cv2.putText(frame, "Backend Logic", (panel[0] + 22, panel[1] + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (240, 248, 250), 2, cv2.LINE_AA)
        active = next((target for target in targets if str(target.get("name", "")) == str(active_target)), None)
        stage_line, detail_line = self._logic_summary(active_stage=active_stage, active=active, mission_metrics=mission_metrics)
        lines = [
            f"active target: {active_target or 'none'}",
            stage_line,
            detail_line,
        ]
        if active is not None:
            lines.extend(
                [
                    f"innovation: {float(active.get('innovation_m', 0.0)):0.2f} m",
                    f"spoof gate: {'ON' if bool(active.get('spoofing_detected', False)) else 'OFF'}",
                    f"packet hold: {'YES' if bool(active.get('packet_dropped', False)) else 'NO'}",
                ]
            )
        if mission_metrics is not None:
            lines.extend(
                [
                    f"rmse: {float(mission_metrics.get('rmse_m', 0.0)):0.2f} m",
                    f"safe intercepts: {int(mission_metrics.get('safe_intercepts', 0))}",
                ]
            )
        for index, line in enumerate(lines):
            cv2.putText(frame, line, (panel[0] + 24, panel[1] + 78 + index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (198, 232, 238), 1, cv2.LINE_AA)

    def _draw_tactical_inset(
        self,
        frame: np.ndarray,
        targets: list[dict[str, Any]],
        interceptor: np.ndarray,
        active_target: str | None,
    ) -> None:
        panel = (1240, 450, 1860, 880)
        cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (10, 18, 28), -1)
        cv2.rectangle(frame, (panel[0], panel[1]), (panel[2], panel[3]), (90, 216, 194), 1, cv2.LINE_AA)
        cv2.putText(frame, "Tactical Inset", (panel[0] + 22, panel[1] + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (240, 248, 250), 2, cv2.LINE_AA)
        cx = (panel[0] + panel[2]) // 2
        cy = (panel[1] + panel[3]) // 2 + 24
        cv2.circle(frame, (cx, cy), 120, (34, 68, 82), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 210, (28, 52, 64), 1, cv2.LINE_AA)
        cv2.line(frame, (panel[0] + 32, cy), (panel[2] - 32, cy), (24, 46, 58), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, panel[1] + 56), (cx, panel[3] - 28), (24, 46, 58), 1, cv2.LINE_AA)
        self._draw_interceptor_vehicle(frame, center=(cx, cy))
        for target in targets:
            relative = np.asarray(target.get("position", interceptor), dtype=float) - interceptor
            inset_x = int(np.clip(cx + relative[1] * 1.4, panel[0] + 40, panel[2] - 40))
            inset_y = int(np.clip(cy - relative[0] * 0.9, panel[1] + 64, panel[3] - 40))
            color = (84, 255, 176) if not bool(target.get("jammed", False)) else (76, 76, 255)
            cv2.circle(frame, (inset_x, inset_y), 9, color, -1, cv2.LINE_AA)
            if bool(target.get("jammed", False)):
                self._draw_hit_marker(frame, center=(inset_x, inset_y), scale=0.55, fresh=bool(target.get("jam_transition", False)))
            if str(target.get("name", "")) == str(active_target):
                cv2.circle(frame, (inset_x, inset_y), 18, (255, 214, 92), 2, cv2.LINE_AA)
            cv2.putText(frame, str(target.get("name", "Target")), (inset_x + 12, inset_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 236, 242), 1, cv2.LINE_AA)
        cv2.putText(frame, "interceptor centered", (panel[0] + 24, panel[3] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 226), 1, cv2.LINE_AA)

    def _logic_summary(
        self,
        active_stage: str | None,
        active: dict[str, Any] | None,
        mission_metrics: dict[str, Any] | None,
    ) -> tuple[str, str]:
        stage = (active_stage or "tracking").lower()
        if active is not None and bool(active.get("jam_transition", False)):
            return (
                "kill radius reached on the active target",
                "the target transitions to jammed and the replay marks the hit event",
            )
        if active is not None and bool(active.get("spoofing_detected", False)):
            return (
                "ekf innovation gating rejected the spoofed fix",
                "navigation falls back to the filtered estimate instead of the raw measurement",
            )
        if active is not None and bool(active.get("packet_dropped", False)):
            return (
                "telemetry packet loss is active on this track",
                "the controller is riding the buffered estimate until the next valid update arrives",
            )
        if "terminal" in stage:
            return (
                "terminal guidance is tightening the intercept arc",
                "closing speed is traded for precision so the interceptor does not overshoot the target",
            )
        if "spoof" in stage or "track" in stage:
            return (
                "target selection follows backend threat ranking",
                "the highest-threat target is locked while the estimator suppresses spoofing noise",
            )
        rmse_value = 0.0 if mission_metrics is None else float(mission_metrics.get("rmse_m", 0.0))
        return (
            "guidance is still shaping the pursuit geometry",
            f"current replay rmse is {rmse_value:0.2f} m across the tracked target set",
        )


__all__ = ["CinematicFrame", "CinematicRecorder"]
