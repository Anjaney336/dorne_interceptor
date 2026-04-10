from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion, simulate_gps_with_drift
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.planning.fallback import FallbackWaypointPlanner
from drone_interceptor.planning.planner import InterceptPlanner
from drone_interceptor.prediction.predictor import TargetPredictor
from drone_interceptor.ros2.px4_bridge import PX4OffboardCommandAdapter, PX4SITLAdapter
from drone_interceptor.simulation.airsim_control import AirSimInterceptorAdapter
from drone_interceptor.tracking.tracker import TargetTracker
from drone_interceptor.types import ControlCommand, Detection, NavigationState, Plan, SensorPacket, TargetState


@dataclass(slots=True)
class TopicEnvelope:
    topic: str
    payload: dict[str, Any]
    timestamp: float


class LocalTopicBus:
    """Minimal in-process topic bus that mirrors ROS2-style publish/subscribe flows."""

    def __init__(self) -> None:
        self._latest: dict[str, TopicEnvelope] = {}
        self._history: dict[str, list[TopicEnvelope]] = {}

    def publish(self, topic: str, payload: dict[str, Any], timestamp: float) -> None:
        envelope = TopicEnvelope(topic=topic, payload=payload, timestamp=float(timestamp))
        self._latest[topic] = envelope
        self._history.setdefault(topic, []).append(envelope)

    def latest(self, topic: str) -> dict[str, Any] | None:
        envelope = self._latest.get(topic)
        return None if envelope is None else envelope.payload

    def history(self, topic: str) -> list[TopicEnvelope]:
        return list(self._history.get(topic, ()))


@dataclass(slots=True)
class NodeStats:
    name: str
    publishes: int = 0
    total_runtime_s: float = 0.0

    @property
    def fps(self) -> float:
        if self.total_runtime_s <= 0.0:
            return 0.0
        return float(self.publishes / self.total_runtime_s)


@dataclass(slots=True)
class EdgeProfile:
    enabled: bool = False
    detection_stride: int = 1
    injected_latency_s: float = 0.0
    inference_imgsz: int | None = None


@dataclass(slots=True)
class ControlCycleResult:
    plan: Plan
    command: ControlCommand
    command_payload: dict[str, Any]
    fallback_used: bool
    px4_setpoint: dict[str, Any]
    airsim_mode: str


@dataclass(frozen=True, slots=True)
class LatencyBudget:
    perception_ms: float
    tracking_ms: float
    navigation_ms: float
    control_ms: float

    @property
    def end_to_end_ms(self) -> float:
        return float(self.perception_ms + self.tracking_ms + self.navigation_ms + self.control_ms)


class _BaseLocalNode:
    def __init__(self, name: str, bus: LocalTopicBus) -> None:
        self.name = name
        self.bus = bus
        self.stats = NodeStats(name=name)

    def _record_runtime(self, started_at: float) -> None:
        self.stats.publishes += 1
        self.stats.total_runtime_s += max(time.perf_counter() - started_at, 1e-6)

    def _publish(self, topic: str, payload: dict[str, Any], timestamp: float) -> dict[str, Any]:
        self.bus.publish(topic=topic, payload=payload, timestamp=timestamp)
        return payload


class LocalPerceptionNode(_BaseLocalNode):
    def __init__(self, config: dict[str, Any], bus: LocalTopicBus, edge_profile: EdgeProfile | None = None) -> None:
        super().__init__(name="perception_node", bus=bus)
        local_config = copy.deepcopy(config)
        self._edge_profile = edge_profile or EdgeProfile()
        if self._edge_profile.inference_imgsz is not None:
            local_config.setdefault("perception", {})["inference_imgsz"] = int(self._edge_profile.inference_imgsz)
        self._detector = TargetDetector(local_config)
        self._last_detection: Detection | None = None

    def process(self, observation: dict[str, Any], step: int) -> dict[str, Any]:
        started_at = time.perf_counter()
        time_s = float(np.asarray(observation.get("time", [0.0]), dtype=float)[0])

        if self._edge_profile.enabled and self._edge_profile.injected_latency_s > 0.0:
            time.sleep(self._edge_profile.injected_latency_s)

        if self._edge_profile.enabled and self._edge_profile.detection_stride > 1 and step % self._edge_profile.detection_stride != 0 and self._last_detection is not None:
            detection = Detection(
                position=self._last_detection.position.copy(),
                confidence=self._last_detection.confidence,
                metadata={**self._last_detection.metadata, "edge_reused_detection": True},
                timestamp=time_s,
            )
        else:
            detection = self._detector.detect(observation)
            self._last_detection = detection

        payload = self._publish(
            topic="interceptor/perception/detections",
            payload={
                "position": np.asarray(detection.position, dtype=float).tolist(),
                "confidence": float(detection.confidence),
                "timestamp": float(detection.timestamp if detection.timestamp is not None else time_s),
                "metadata": dict(detection.metadata),
            },
            timestamp=time_s,
        )
        self._record_runtime(started_at)
        return payload


class LocalTrackingNode(_BaseLocalNode):
    def __init__(self, config: dict[str, Any], bus: LocalTopicBus) -> None:
        super().__init__(name="tracking_node", bus=bus)
        self._tracker = TargetTracker(config)

    def process(self, detection_payload: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        detection = Detection(
            position=np.asarray(detection_payload["position"], dtype=float),
            confidence=float(detection_payload["confidence"]),
            metadata=dict(detection_payload.get("metadata", {})),
            timestamp=float(detection_payload.get("timestamp", 0.0)),
        )
        track = self._tracker.update(detection)
        payload = self._publish(
            topic="interceptor/tracking/state",
            payload={
                "position": np.asarray(track.position, dtype=float).tolist(),
                "velocity": np.asarray(track.velocity, dtype=float).tolist(),
                "acceleration": (
                    np.asarray(track.acceleration, dtype=float).tolist()
                    if track.acceleration is not None
                    else [0.0, 0.0, 0.0]
                ),
                "covariance": None if track.covariance is None else np.asarray(track.covariance, dtype=float).tolist(),
                "timestamp": float(track.timestamp if track.timestamp is not None else detection.timestamp or 0.0),
                "track_id": track.track_id,
                "metadata": dict(track.metadata),
            },
            timestamp=float(track.timestamp if track.timestamp is not None else detection.timestamp or 0.0),
        )
        self._record_runtime(started_at)
        return payload


class LocalNavigationNode(_BaseLocalNode):
    def __init__(self, config: dict[str, Any], bus: LocalTopicBus) -> None:
        super().__init__(name="navigation_node", bus=bus)
        self._navigator = GPSIMUKalmanFusion(config)
        self._drift_rate_mps = float(config.get("navigation", {}).get("gps_drift_rate_mps", 0.2))

    def process(self, packet: SensorPacket) -> dict[str, Any]:
        started_at = time.perf_counter()
        state = self._navigator.update(packet)
        drifted_position = simulate_gps_with_drift(
            true_position=np.asarray(packet.true_position if packet.true_position is not None else packet.gps_position, dtype=float),
            time_s=float(packet.timestamp),
            drift_rate_mps=self._drift_rate_mps,
        )
        payload = self._publish(
            topic="interceptor/navigation/state",
            payload={
                "position": np.asarray(state.position, dtype=float).tolist(),
                "velocity": np.asarray(state.velocity, dtype=float).tolist(),
                "covariance": None if state.covariance is None else np.asarray(state.covariance, dtype=float).tolist(),
                "timestamp": float(state.timestamp),
                "metadata": dict(state.metadata),
                "drifted_position": np.asarray(drifted_position, dtype=float).tolist(),
            },
            timestamp=float(state.timestamp),
        )
        self._record_runtime(started_at)
        return payload


class LocalControlNode(_BaseLocalNode):
    def __init__(self, config: dict[str, Any], bus: LocalTopicBus, use_airsim: bool = False) -> None:
        super().__init__(name="control_node", bus=bus)
        self._config = config
        self._predictor = TargetPredictor(config)
        self._planner = InterceptPlanner(config)
        self._fallback = FallbackWaypointPlanner(config)
        self._controller = InterceptionController(config)
        self._px4_adapter = PX4OffboardCommandAdapter()
        self._sitl_adapter = PX4SITLAdapter.from_config(config)
        self._airsim_adapter = AirSimInterceptorAdapter.from_config(config, connect=use_airsim)
        planning = config.get("planning", {})
        self._fallback_step = int(
            planning.get(
                "fallback_replan_step",
                max(int(config.get("mission", {}).get("max_steps", 250)) * 0.55, 60),
            )
        )
        self._intercept_threshold = float(config.get("planning", {}).get("desired_intercept_distance_m", 10.0))
        self._replan_margin = float(planning.get("replan_distance_margin_m", 1.5 * self._intercept_threshold))

    def process(
        self,
        navigation_payload: dict[str, Any],
        tracking_payload: dict[str, Any],
        step: int,
        dt: float,
        true_distance_m: float,
    ) -> ControlCycleResult:
        started_at = time.perf_counter()
        interceptor_state = _navigation_payload_to_state(navigation_payload)
        track_state = _tracking_payload_to_state(tracking_payload)
        prediction = self._predictor.predict(track_state)
        plan = self._planner.plan(interceptor_state, prediction)
        true_position = tracking_payload.get("metadata", {}).get("true_position")
        tracking_error_m = 0.0
        if true_position is not None:
            tracking_error_m = float(
                np.linalg.norm(np.asarray(track_state.position, dtype=float) - np.asarray(true_position, dtype=float))
            )
        elif track_state.covariance is not None:
            tracking_error_m = float(np.sqrt(max(np.trace(np.asarray(track_state.covariance, dtype=float)), 0.0)))
        plan.metadata["current_target_position"] = np.asarray(track_state.position, dtype=float).copy()
        plan.metadata["current_target_velocity"] = np.asarray(track_state.velocity, dtype=float).copy()
        plan.metadata["current_target_acceleration"] = (
            np.asarray(track_state.acceleration, dtype=float).copy()
            if track_state.acceleration is not None
            else np.zeros(3, dtype=float)
        )
        plan.metadata["current_target_covariance"] = (
            None if track_state.covariance is None else np.asarray(track_state.covariance, dtype=float).copy()
        )
        plan.metadata["tracking_error_m"] = tracking_error_m

        fallback_used = bool(step >= self._fallback_step and true_distance_m > self._replan_margin)
        if fallback_used:
            plan = self._fallback.plan(
                interceptor_state=interceptor_state,
                current_target_state=track_state,
                predicted_target_states=prediction,
            )
            plan.metadata["current_target_position"] = np.asarray(track_state.position, dtype=float).copy()
            plan.metadata["current_target_velocity"] = np.asarray(track_state.velocity, dtype=float).copy()
            plan.metadata["current_target_acceleration"] = (
                np.asarray(track_state.acceleration, dtype=float).copy()
                if track_state.acceleration is not None
                else np.zeros(3, dtype=float)
            )
            plan.metadata["current_target_covariance"] = (
                None if track_state.covariance is None else np.asarray(track_state.covariance, dtype=float).copy()
            )
            plan.metadata["tracking_error_m"] = tracking_error_m

        command = self._controller.compute_command(interceptor_state, plan)
        px4_setpoint = self._px4_adapter.to_offboard_setpoint(command)
        sitl_packet = self._sitl_adapter.command_packet(px4_setpoint)
        airsim_packet = self._airsim_adapter.dispatch(
            command=command,
            altitude_m=float(interceptor_state.position[2]),
            dt=dt,
        )

        payload = self._publish(
            topic="interceptor/control/command",
            payload={
                "velocity_command": np.asarray(command.velocity_command, dtype=float).tolist(),
                "acceleration_command": (
                    np.asarray(command.acceleration_command, dtype=float).tolist()
                    if command.acceleration_command is not None
                    else [0.0, 0.0, 0.0]
                ),
                "mode": command.mode,
                "metadata": dict(command.metadata),
                "px4_setpoint": px4_setpoint,
                "px4_mode": sitl_packet.mode,
                "px4_sitl_command": sitl_packet.sitl_command,
                "fallback_used": fallback_used,
            },
            timestamp=float(navigation_payload.get("timestamp", 0.0)),
        )
        self._record_runtime(started_at)
        return ControlCycleResult(
            plan=plan,
            command=command,
            command_payload=payload,
            fallback_used=fallback_used,
            px4_setpoint=px4_setpoint,
            airsim_mode=airsim_packet.mode,
        )


def _navigation_payload_to_state(payload: dict[str, Any]) -> NavigationState:
    covariance = payload.get("covariance")
    return NavigationState(
        position=np.asarray(payload["position"], dtype=float),
        velocity=np.asarray(payload["velocity"], dtype=float),
        covariance=None if covariance is None else np.asarray(covariance, dtype=float),
        timestamp=float(payload.get("timestamp", 0.0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _tracking_payload_to_state(payload: dict[str, Any]) -> TargetState:
    covariance = payload.get("covariance")
    return TargetState(
        position=np.asarray(payload["position"], dtype=float),
        velocity=np.asarray(payload["velocity"], dtype=float),
        acceleration=np.asarray(payload.get("acceleration", [0.0, 0.0, 0.0]), dtype=float),
        covariance=None if covariance is None else np.asarray(covariance, dtype=float),
        timestamp=float(payload.get("timestamp", 0.0)),
        track_id=payload.get("track_id"),
        metadata=dict(payload.get("metadata", {})),
    )


__all__ = [
    "ControlCycleResult",
    "EdgeProfile",
    "LatencyBudget",
    "LocalControlNode",
    "LocalNavigationNode",
    "LocalPerceptionNode",
    "LocalTopicBus",
    "LocalTrackingNode",
    "NodeStats",
    "TopicEnvelope",
    "build_detection_message",
    "build_latency_budget_report",
    "build_mission_command_message",
    "build_navigation_message",
    "build_prediction_message",
    "build_spoofing_alert_message",
    "build_tracking_message",
]


def build_detection_message(position: np.ndarray, confidence: float, timestamp: float, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "interceptor/perception/detections",
        "position": np.asarray(position, dtype=float).tolist(),
        "confidence": float(confidence),
        "timestamp": float(timestamp),
        "metadata": dict(metadata),
    }


def build_tracking_message(track_state: TargetState) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "interceptor/tracking/state",
        "position": np.asarray(track_state.position, dtype=float).tolist(),
        "velocity": np.asarray(track_state.velocity, dtype=float).tolist(),
        "acceleration": (
            np.asarray(track_state.acceleration, dtype=float).tolist()
            if track_state.acceleration is not None
            else [0.0, 0.0, 0.0]
        ),
        "timestamp": float(track_state.timestamp if track_state.timestamp is not None else 0.0),
        "track_id": track_state.track_id,
        "metadata": dict(track_state.metadata),
    }


def build_prediction_message(prediction: list[TargetState], timestamp: float) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "interceptor/prediction/trajectory",
        "timestamp": float(timestamp),
        "states": [
            {
                "position": np.asarray(state.position, dtype=float).tolist(),
                "velocity": np.asarray(state.velocity, dtype=float).tolist(),
                "track_id": state.track_id,
                "metadata": dict(state.metadata),
            }
            for state in prediction
        ],
    }


def build_navigation_message(state: NavigationState, drifted_position: np.ndarray | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "interceptor/navigation/state",
        "position": np.asarray(state.position, dtype=float).tolist(),
        "velocity": np.asarray(state.velocity, dtype=float).tolist(),
        "covariance": None if state.covariance is None else np.asarray(state.covariance, dtype=float).tolist(),
        "timestamp": float(state.timestamp),
        "metadata": dict(state.metadata),
        "drifted_position": None if drifted_position is None else np.asarray(drifted_position, dtype=float).tolist(),
    }


def build_spoofing_alert_message(timestamp: float, detected: bool, innovation_m: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "interceptor/navigation/spoofing_alert",
        "timestamp": float(timestamp),
        "detected": bool(detected),
        "innovation_m": float(innovation_m),
        "metadata": {} if metadata is None else dict(metadata),
    }


def build_mission_command_message(command: ControlCommand, timestamp: float, px4_setpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topic": "interceptor/control/command",
        "timestamp": float(timestamp),
        "velocity_command": np.asarray(command.velocity_command, dtype=float).tolist(),
        "acceleration_command": (
            np.asarray(command.acceleration_command, dtype=float).tolist()
            if command.acceleration_command is not None
            else [0.0, 0.0, 0.0]
        ),
        "mode": command.mode,
        "metadata": dict(command.metadata),
        "px4_setpoint": {} if px4_setpoint is None else dict(px4_setpoint),
    }


def build_latency_budget_report(
    perception_stats: NodeStats,
    tracking_stats: NodeStats,
    navigation_stats: NodeStats,
    control_stats: NodeStats,
) -> dict[str, Any]:
    budget = LatencyBudget(
        perception_ms=_stats_latency_ms(perception_stats),
        tracking_ms=_stats_latency_ms(tracking_stats),
        navigation_ms=_stats_latency_ms(navigation_stats),
        control_ms=_stats_latency_ms(control_stats),
    )
    return {
        "schema_version": "1.0",
        "perception_ms": budget.perception_ms,
        "tracking_ms": budget.tracking_ms,
        "navigation_ms": budget.navigation_ms,
        "control_ms": budget.control_ms,
        "end_to_end_ms": budget.end_to_end_ms,
    }


def _stats_latency_ms(stats: NodeStats) -> float:
    if stats.publishes <= 0:
        return 0.0
    return float((stats.total_runtime_s / stats.publishes) * 1000.0)
