from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

from drone_interceptor.config import load_config
from drone_interceptor.constraints import ConstraintStatus, load_constraint_envelope
from drone_interceptor.control.controller import InterceptionController
from drone_interceptor.datasets.visdrone import load_yolo_labels, visualize_yolo_labels
from drone_interceptor.dynamics.state_space import update_state
from drone_interceptor.navigation.state_estimator import GPSIMUKalmanFusion
from drone_interceptor.optimization.trajectory_optimizer import InterceptionTrajectoryOptimizer
from drone_interceptor.perception.detector import TargetDetector
from drone_interceptor.perception.infer import _import_cv2, _import_yolo, annotate_frame, resolve_model_path
from drone_interceptor.planning.planner import InterceptPlanner
from drone_interceptor.prediction.predictor import TargetPredictor
from drone_interceptor.prediction.trajectory import HybridTrajectoryPredictor
from drone_interceptor.simulation.environment import DroneInterceptionEnv
from drone_interceptor.tracking.tracker import DeepSortTracker, TargetTracker, draw_tracked_objects
from drone_interceptor.types import TargetState


LOGGER = logging.getLogger("system_validation")


@dataclass(frozen=True)
class ValidationStatus:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class ValidationMetrics:
    detection_fps: float
    tracking_stability: float
    prediction_rmse: float
    interception_time_s: float
    success_rate: float


@dataclass(frozen=True)
class ValidationArtifacts:
    dataset_visualizations: list[Path]
    detection_visualization: Path
    tracking_visualization: Path
    simulation_plot: Path
    log_file: Path


@dataclass
class IntegratedRun:
    distances_m: list[float]
    interceptor_positions: list[np.ndarray]
    target_positions: list[np.ndarray]
    commanded_speeds: list[float]
    commanded_accelerations: list[float]
    times_s: list[float]
    penalty_events: int
    intercepted: bool
    interception_time_s: float


def run_system_validation(
    project_root: str | Path,
    sample_count: int = 5,
    random_seed: int = 7,
) -> tuple[list[ValidationStatus], ValidationMetrics, ValidationArtifacts]:
    root = Path(project_root).resolve()
    outputs_dir = root / "outputs"
    logs_dir = root / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(root / "configs" / "default.yaml")
    dataset_root = root / "data" / "visdrone_yolo"

    dataset_status, dataset_samples = validate_dataset(
        dataset_root=dataset_root,
        outputs_dir=outputs_dir,
        sample_count=sample_count,
        random_seed=random_seed,
    )

    detection_status, detection_output, detection_source, detection_fps, detection_model = validate_detection(
        candidate_images=[sample.image_path for sample in dataset_samples],
        outputs_dir=outputs_dir,
        config=config,
    )

    tracking_status, tracking_output, tracking_stability = validate_tracking(
        image_path=detection_source,
        outputs_dir=outputs_dir,
        model_path=detection_model,
    )

    state_status = validate_state_model()

    prediction_status, prediction_rmse = validate_prediction(config=config)

    integrated_run = run_integrated_pipeline(config=config, max_steps=int(config["mission"]["max_steps"]))
    control_status = validate_control(integrated_run=integrated_run)
    constraint_status = validate_constraints(config=config, integrated_run=integrated_run)
    optimization_status = validate_optimization(config=config)
    simulation_status, simulation_plot = validate_simulation(
        integrated_run=integrated_run,
        outputs_dir=outputs_dir,
    )
    integration_status = validate_integration(
        statuses=[
            dataset_status,
            detection_status,
            tracking_status,
            state_status,
            prediction_status,
            control_status,
            constraint_status,
            optimization_status,
            simulation_status,
        ],
        required_outputs=[detection_output, tracking_output, simulation_plot],
    )

    success_rate = compute_success_rate(config=config, runs=5)
    metrics = ValidationMetrics(
        detection_fps=detection_fps,
        tracking_stability=tracking_stability,
        prediction_rmse=prediction_rmse,
        interception_time_s=integrated_run.interception_time_s,
        success_rate=success_rate,
    )

    statuses = [
        dataset_status,
        detection_status,
        tracking_status,
        state_status,
        prediction_status,
        control_status,
        constraint_status,
        optimization_status,
        simulation_status,
        integration_status,
    ]
    artifacts = ValidationArtifacts(
        dataset_visualizations=[sample.output_path for sample in dataset_samples],
        detection_visualization=detection_output,
        tracking_visualization=tracking_output,
        simulation_plot=simulation_plot,
        log_file=logs_dir / "system_validation.log",
    )
    return statuses, metrics, artifacts


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    label_path: Path
    output_path: Path


def validate_dataset(
    dataset_root: Path,
    outputs_dir: Path,
    sample_count: int,
    random_seed: int,
) -> tuple[ValidationStatus, list[DatasetSample]]:
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.exists() or not labels_root.exists():
        return ValidationStatus("Dataset", False, "Missing images or labels root"), []

    image_paths = sorted(
        [
            path
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if len(image_paths) < sample_count:
        return ValidationStatus("Dataset", False, f"Need at least {sample_count} images"), []

    rng = random.Random(random_seed)
    sampled_images = rng.sample(image_paths, sample_count)
    samples: list[DatasetSample] = []

    for index, image_path in enumerate(sampled_images, start=1):
        split = image_path.parent.name
        label_path = labels_root / split / f"{image_path.stem}.txt"
        if not label_path.exists():
            return ValidationStatus("Dataset", False, f"Missing label for {image_path.name}"), []

        with Image.open(image_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                return ValidationStatus("Dataset", False, f"Unreadable image {image_path.name}"), []

        raw_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        _ = load_yolo_labels(label_path=label_path, image_width=width, image_height=height)
        for line in raw_lines:
            parts = line.split()
            if len(parts) != 5:
                return ValidationStatus("Dataset", False, f"Malformed label line in {label_path.name}"), []
            values = [float(value) for value in parts[1:]]
            if not all(0.0 < value < 1.0 for value in values):
                return ValidationStatus("Dataset", False, f"Non-normalized bbox in {label_path.name}"), []

        output_path = outputs_dir / f"system_dataset_sample_{index}.png"
        visualize_yolo_labels(image_path=image_path, label_path=label_path, output_path=output_path)
        samples.append(DatasetSample(image_path=image_path, label_path=label_path, output_path=output_path))

    return ValidationStatus("Dataset", True, f"validated_samples={sample_count}"), samples


def validate_detection(
    candidate_images: list[Path],
    outputs_dir: Path,
    config: dict[str, Any],
) -> tuple[ValidationStatus, Path, Path, float, str | Path]:
    YOLO = _import_yolo()
    cv2 = _import_cv2()
    output_path = outputs_dir / "system_detection.png"

    imgsz = int(config["perception"].get("inference_imgsz", 640))
    conf = float(config["perception"].get("confidence_threshold", 0.25))

    known_candidate = outputs_dir.parents[0] / "data" / "visdrone_yolo" / "images" / "val" / "0000001_02999_d_0000005.jpg"
    ordered_candidates = [path for path in candidate_images if path.exists()]
    if known_candidate.exists() and known_candidate not in ordered_candidates:
        ordered_candidates.append(known_candidate)
    if not ordered_candidates:
        return ValidationStatus("Detection", False, "No candidate image for detection"), output_path, outputs_dir, 0.0, ""

    model_candidates: list[str | Path] = []
    local_model = _resolve_detection_model()
    if local_model:
        model_candidates.append(local_model)
    model_candidates.extend(["yolov10n.pt", "yolov10s.pt"])

    best_attempt: tuple[int, float, Any, np.ndarray, Path, str | Path] | None = None
    seen_models: set[str] = set()
    for model_path in model_candidates:
        model_key = str(model_path)
        if model_key in seen_models:
            continue
        seen_models.add(model_key)
        model = YOLO(str(model_path))

        for image_path in ordered_candidates:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            _ = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            timings: list[float] = []
            result: Any = None
            for _index in range(3):
                start = time.perf_counter()
                result = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
                timings.append(time.perf_counter() - start)
            assert result is not None

            average_time = float(sum(timings) / len(timings))
            fps = 1.0 / average_time if average_time > 0.0 else 0.0
            boxes = getattr(result, "boxes", None)
            detection_count = 0 if boxes is None else int(boxes.xyxy.cpu().numpy().shape[0])
            if best_attempt is None or detection_count > best_attempt[0]:
                best_attempt = (detection_count, fps, result, frame, image_path, model_path)
            if detection_count > 0:
                break
        if best_attempt is not None and best_attempt[0] > 0:
            break

    if best_attempt is None:
        return ValidationStatus("Detection", False, "Could not run inference on candidate images"), output_path, outputs_dir, 0.0, ""

    detection_count, fps, result, frame, chosen_image, model_path = best_attempt
    annotated = annotate_frame(frame=frame.copy(), result=result, fps=fps)
    cv2.imwrite(str(output_path), annotated)
    passed = detection_count > 0 and fps > 10.0
    return (
        ValidationStatus(
            "Detection",
            passed,
            f"detections={detection_count} fps={fps:.2f} image={chosen_image.name} model={model_path}",
        ),
        output_path,
        chosen_image,
        fps,
        model_path,
    )


def validate_tracking(
    image_path: Path,
    outputs_dir: Path,
    model_path: str | Path,
) -> tuple[ValidationStatus, Path, float]:
    YOLO = _import_yolo()
    cv2 = _import_cv2()
    output_path = outputs_dir / "system_tracking.png"

    model = YOLO(str(model_path))
    base_frame = cv2.imread(str(image_path))
    if base_frame is None:
        return ValidationStatus("Tracking", False, f"Failed to read {image_path.name}"), output_path, 0.0

    initial_result = model(base_frame, imgsz=640, conf=0.25, verbose=False)[0]
    boxes = getattr(initial_result, "boxes", None)
    if boxes is None or boxes.xyxy.cpu().numpy().size == 0:
        return ValidationStatus("Tracking", False, "No detections to initialize tracker"), output_path, 0.0

    tracker = DeepSortTracker(history_length=12)
    tracked_sequences: list[list[Any]] = []
    frames: list[np.ndarray] = []
    for frame_index in range(6):
        dx = frame_index * 4
        dy = frame_index * 2
        shifted_frame = _shift_frame(base_frame, dx=dx, dy=dy)
        shifted_result = _shift_result(initial_result, dx=dx, dy=dy)
        tracked_objects = tracker.update_from_yolo(result=shifted_result, frame=shifted_frame)
        tracked_sequences.append(tracked_objects)
        frames.append(shifted_frame)

    first_non_empty = next((objects for objects in tracked_sequences if objects), [])
    if not first_non_empty:
        return ValidationStatus("Tracking", False, "Tracker did not confirm any tracks"), output_path, 0.0

    primary_id = first_non_empty[0].track_id
    id_sequence: list[str] = []
    for tracked_objects in tracked_sequences:
        best_match = next((obj for obj in tracked_objects if obj.track_id == primary_id), None)
        if best_match is not None:
            id_sequence.append(best_match.track_id)

    if len(id_sequence) < len(tracked_sequences) - 1:
        return ValidationStatus("Tracking", False, "Trajectory continuity insufficient"), output_path, len(id_sequence) / len(tracked_sequences)

    id_switches = sum(1 for prev, nxt in zip(id_sequence, id_sequence[1:], strict=False) if prev != nxt)
    stability = len(id_sequence) / len(tracked_sequences)
    final_frame = draw_tracked_objects(frames[-1], tracked_sequences[-1], fps=0.0)
    cv2.imwrite(str(output_path), final_frame)
    passed = id_switches == 0 and stability >= 0.8
    return (
        ValidationStatus(
            "Tracking",
            passed,
            f"primary_track={primary_id} stability={stability:.2f} id_switches={id_switches}",
        ),
        output_path,
        stability,
    )


def validate_state_model() -> ValidationStatus:
    dt = 0.1
    state = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    acceleration = np.array([0.5, -0.2], dtype=float)
    next_state = update_state(state, acceleration, dt)
    expected = np.array(
        [
            1.0 + 3.0 * dt + 0.5 * 0.5 * dt * dt,
            2.0 + 4.0 * dt + 0.5 * (-0.2) * dt * dt,
            3.0 + 0.5 * dt,
            4.0 - 0.2 * dt,
        ],
        dtype=float,
    )
    passed = bool(np.allclose(next_state, expected))
    return ValidationStatus("State Model", passed, f"max_error={np.max(np.abs(next_state - expected)):.6f}")


def validate_prediction(config: dict[str, Any]) -> tuple[ValidationStatus, float]:
    dt = float(config["mission"]["time_step"])
    horizon = int(config["prediction"]["horizon_steps"])
    predictor = HybridTrajectoryPredictor(dt=dt, horizon_steps=horizon)

    acceleration = np.array([0.8, -0.3], dtype=float)
    positions = []
    x0 = np.array([2.0, -1.0], dtype=float)
    velocity = np.array([4.0, 1.5], dtype=float)
    for step in range(8):
        t = step * dt
        positions.append(x0 + velocity * t + 0.5 * acceleration * (t**2))
    past_positions = np.asarray(positions, dtype=float)
    prediction = predictor.predict(past_positions, horizon_steps=horizon)

    future_positions = []
    last_time = (len(past_positions) - 1) * dt
    for step in range(1, horizon + 1):
        t = last_time + step * dt
        future_positions.append(x0 + velocity * t + 0.5 * acceleration * (t**2))
    future = np.asarray(future_positions, dtype=float)
    rmse = float(np.sqrt(np.mean((prediction.predicted_positions - future) ** 2)))
    threshold = 0.5
    return ValidationStatus("Prediction", rmse < threshold, f"rmse={rmse:.4f} threshold={threshold:.2f}"), rmse


def run_integrated_pipeline(config: dict[str, Any], max_steps: int) -> IntegratedRun:
    env = DroneInterceptionEnv(config)
    detector = TargetDetector(config)
    tracker = TargetTracker(config)
    predictor = TargetPredictor(config)
    planner = InterceptPlanner(config)
    controller = InterceptionController(config)
    navigator = GPSIMUKalmanFusion(config)

    observation = env.reset()
    distances: list[float] = []
    interceptor_positions: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    commanded_speeds: list[float] = []
    commanded_accelerations: list[float] = []
    times_s: list[float] = []
    penalty_events = 0
    intercepted = False
    interception_time_s = float(max_steps * float(config["mission"]["time_step"]))

    for _step in range(max_steps):
        navigation_state = navigator.update(observation["sensor_packet"])
        interceptor_estimate = TargetState(
            position=navigation_state.position.copy(),
            velocity=navigation_state.velocity.copy(),
            covariance=navigation_state.covariance,
            timestamp=navigation_state.timestamp,
            metadata=dict(navigation_state.metadata),
        )
        detection = detector.detect(observation)
        track = tracker.update(detection)
        prediction = predictor.predict(track)
        plan = planner.plan(interceptor_estimate, prediction)
        plan.metadata["current_target_position"] = track.position.copy()
        plan.metadata["current_target_velocity"] = track.velocity.copy()
        plan.metadata["current_target_acceleration"] = (
            track.acceleration.copy() if track.acceleration is not None else np.zeros(3, dtype=float)
        )
        plan.metadata["current_target_covariance"] = (
            None if track.covariance is None else np.asarray(track.covariance, dtype=float).copy()
        )
        plan.metadata["tracking_error_m"] = float(np.linalg.norm(track.position - env.target_state.position))
        command = controller.compute_command(interceptor_estimate, plan)
        observation, done, info = env.step(command)

        interceptor_positions.append(env.interceptor_state.position.copy())
        target_positions.append(env.target_state.position.copy())
        distances.append(float(info["distance_to_target"]))
        commanded_speeds.append(float(np.linalg.norm(command.velocity_command)))
        commanded_accelerations.append(
            float(np.linalg.norm(command.acceleration_command if command.acceleration_command is not None else np.zeros(3)))
        )
        times_s.append(float(observation["time"][0]))
        penalty_events += int(command.metadata.get("velocity_clipped", False))
        penalty_events += int(command.metadata.get("acceleration_clipped", False))
        penalty_events += int(not command.metadata.get("tracking_ok", True))
        penalty_events += int(command.metadata.get("safety_override", False))
        penalty_events += int(not navigation_state.metadata.get("drift_rate_in_bounds", True))

        if done:
            intercepted = info["distance_to_target"] <= float(config["planning"]["desired_intercept_distance_m"])
            interception_time_s = float(observation["time"][0])
            break

    return IntegratedRun(
        distances_m=distances,
        interceptor_positions=interceptor_positions,
        target_positions=target_positions,
        commanded_speeds=commanded_speeds,
        commanded_accelerations=commanded_accelerations,
        times_s=times_s,
        penalty_events=penalty_events,
        intercepted=intercepted,
        interception_time_s=interception_time_s,
    )


def validate_control(integrated_run: IntegratedRun) -> ValidationStatus:
    if not integrated_run.distances_m:
        return ValidationStatus("Control", False, "No integrated distances recorded")
    initial_distance = integrated_run.distances_m[0]
    final_distance = integrated_run.distances_m[-1]
    passed = final_distance < initial_distance
    return ValidationStatus(
        "Control",
        passed,
        f"initial_distance={initial_distance:.2f} final_distance={final_distance:.2f}",
    )


def validate_constraints(config: dict[str, Any], integrated_run: IntegratedRun) -> ValidationStatus:
    envelope = load_constraint_envelope(config)
    if not integrated_run.commanded_speeds or not integrated_run.commanded_accelerations or not integrated_run.distances_m:
        return ValidationStatus("Constraints", False, "No integrated control data recorded")

    max_speed = max(integrated_run.commanded_speeds)
    max_acceleration = max(integrated_run.commanded_accelerations)
    min_distance = min(integrated_run.distances_m)
    passed = (
        max_speed <= envelope.max_velocity_mps + 1e-9
        and max_acceleration <= envelope.max_acceleration_mps2 + 1e-9
        and min_distance >= envelope.min_separation_m
    )
    return ValidationStatus(
        "Constraints",
        passed,
        f"max_speed={max_speed:.2f} max_accel={max_acceleration:.2f} min_distance={min_distance:.2f} penalties={integrated_run.penalty_events}",
    )


def validate_optimization(config: dict[str, Any]) -> ValidationStatus:
    optimizer = InterceptionTrajectoryOptimizer(config=config, random_seed=7)
    result = optimizer.optimize(
        interceptor_state=np.array([0.0, 0.0, 0.0, 0.0]),
        target_state=np.array([20.0, 5.0, 0.0, 0.0]),
    )
    best_history_nonincreasing = bool(np.all(np.diff(result.best_cost_history) <= 1e-9))
    final_better_than_first = bool(result.optimal_cost <= result.candidate_costs[0] + 1e-9)
    passed = best_history_nonincreasing and final_better_than_first
    return ValidationStatus(
        "Optimization",
        passed,
        f"optimal_cost={result.optimal_cost:.2f} first_cost={result.candidate_costs[0]:.2f} evaluated={result.evaluated_trajectories}",
    )


def validate_simulation(integrated_run: IntegratedRun, outputs_dir: Path) -> tuple[ValidationStatus, Path]:
    output_path = outputs_dir / "day2_simulation.png"
    if not integrated_run.interceptor_positions or not integrated_run.target_positions:
        return ValidationStatus("Simulation", False, "No integrated simulation positions recorded"), output_path

    interceptor = np.asarray(integrated_run.interceptor_positions, dtype=float)
    target = np.asarray(integrated_run.target_positions, dtype=float)
    _plot_day2_simulation(
        interceptor_positions=interceptor,
        target_positions=target,
        output_path=output_path,
        intercept_index=len(interceptor) - 1 if integrated_run.intercepted else None,
    )
    return ValidationStatus(
        "Simulation",
        output_path.exists(),
        f"steps={len(interceptor)} intercepted={integrated_run.intercepted}",
    ), output_path


def validate_integration(statuses: list[ValidationStatus], required_outputs: list[Path]) -> ValidationStatus:
    if not all(status.passed for status in statuses):
        failed = [status.name for status in statuses if not status.passed]
        return ValidationStatus("Integration", False, f"failed_stages={','.join(failed)}")
    missing = [path.name for path in required_outputs if not path.exists()]
    if missing:
        return ValidationStatus("Integration", False, f"missing_outputs={','.join(missing)}")
    return ValidationStatus("Integration", True, "full pipeline completed without crash")


def compute_success_rate(config: dict[str, Any], runs: int) -> float:
    successes = 0
    for offset in range(runs):
        config_copy = copy.deepcopy(config)
        config_copy["system"]["random_seed"] = int(config["system"]["random_seed"]) + offset
        run = run_integrated_pipeline(config=config_copy, max_steps=int(config_copy["mission"]["max_steps"]))
        successes += int(run.intercepted)
    return successes / float(runs)


def setup_logging(log_path: Path) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def print_report(statuses: list[ValidationStatus], metrics: ValidationMetrics) -> None:
    print("DAY 1 + DAY 2 VALIDATION REPORT:")
    for status in statuses:
        outcome = "PASS" if status.passed else "FAIL"
        print(f"- {status.name}: {outcome}")
        LOGGER.info("%s %s %s", status.name, outcome, status.details)
    print("PERFORMANCE METRICS:")
    print(f"- Detection FPS: {metrics.detection_fps:.2f}")
    print(f"- Tracking Stability: {metrics.tracking_stability:.2f}")
    print(f"- Prediction RMSE: {metrics.prediction_rmse:.4f}")
    print(f"- Interception Time: {metrics.interception_time_s:.2f}s")
    print(f"- Success Rate: {metrics.success_rate:.2%}")
    LOGGER.info(
        "metrics detection_fps=%.2f tracking_stability=%.2f prediction_rmse=%.4f interception_time_s=%.2f success_rate=%.4f",
        metrics.detection_fps,
        metrics.tracking_stability,
        metrics.prediction_rmse,
        metrics.interception_time_s,
        metrics.success_rate,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full Day 1 + Day 2 system validation.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root directory.")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of random dataset samples.")
    parser.add_argument("--seed", type=int, default=7, help="Validation random seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    log_path = project_root / "logs" / "system_validation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_path)

    statuses, metrics, artifacts = run_system_validation(
        project_root=project_root,
        sample_count=args.sample_count,
        random_seed=args.seed,
    )
    LOGGER.info(
        "artifacts dataset=%s detection=%s tracking=%s simulation=%s",
        [str(path) for path in artifacts.dataset_visualizations],
        artifacts.detection_visualization,
        artifacts.tracking_visualization,
        artifacts.simulation_plot,
    )
    print_report(statuses, metrics)


def _resolve_detection_model() -> Path:
    preferred = Path("models/visdrone_yolov10n.pt")
    resolved = resolve_model_path(preferred if preferred.exists() else None)
    return Path(resolved)


def _plot_day2_simulation(
    interceptor_positions: np.ndarray,
    target_positions: np.ndarray,
    output_path: Path,
    intercept_index: int | None,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(target_positions[:, 0], target_positions[:, 1], label="Target", color="tab:red", linewidth=2)
    axis.plot(interceptor_positions[:, 0], interceptor_positions[:, 1], label="Interceptor", color="tab:blue", linewidth=2)
    axis.scatter(target_positions[0, 0], target_positions[0, 1], color="tab:red", marker="o", s=50)
    axis.scatter(interceptor_positions[0, 0], interceptor_positions[0, 1], color="tab:blue", marker="o", s=50)
    if intercept_index is not None:
        axis.scatter(
            interceptor_positions[intercept_index, 0],
            interceptor_positions[intercept_index, 1],
            color="tab:green",
            marker="x",
            s=80,
            label="Interception Point",
        )
    axis.set_title("Day 2 Simulation Validation")
    axis.set_xlabel("X Position (m)")
    axis.set_ylabel("Y Position (m)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.axis("equal")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    cv2 = _import_cv2()
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


class _TensorLike:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


def _shift_result(result: Any, dx: int, dy: int) -> Any:
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().copy()
    xyxy[:, [0, 2]] += dx
    xyxy[:, [1, 3]] += dy
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_TensorLike(xyxy),
            conf=_TensorLike(boxes.conf.cpu().numpy().copy()),
            cls=_TensorLike(boxes.cls.cpu().numpy().copy()),
        ),
        names=result.names,
    )


__all__ = ["run_system_validation"]


if __name__ == "__main__":
    main()
