from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JetsonNanoOptimizer:
    """Exports an Ultralytics detector into deployment-ready artifacts."""

    def __init__(self, imgsz: int = 640) -> None:
        self._imgsz = imgsz

    def export(
        self,
        model_path: str | Path,
        output_dir: str | Path,
        export_format: str = "onnx",
        half: bool = True,
        int8: bool = False,
    ) -> Path:
        model = self._load_model(model_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        exported = model.export(
            format=export_format,
            imgsz=self._imgsz,
            half=half,
            int8=int8,
            optimize=True,
        )
        manifest_path = output_path / "jetson_deployment_manifest.json"
        manifest = {
            "source_model": str(model_path),
            "exported_artifact": str(exported),
            "format": export_format,
            "imgsz": self._imgsz,
            "half": half,
            "int8": int8,
            "runtime_notes": [
                "Use TensorRT FP16 on Jetson Nano when available.",
                "Keep input resolution at 640 for stable latency.",
                "Pin detector and tracker to independent threads for real-time throughput.",
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    def recommended_runtime_profile(self) -> dict[str, Any]:
        return {
            "device": "jetson_nano",
            "precision": "fp16",
            "imgsz": self._imgsz,
            "batch": 1,
            "tracker_history_length": 20,
            "notes": "Prefer TensorRT engine export for deployment, with DeepSORT running on CPU.",
        }

    def _load_model(self, model_path: str | Path) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required to export models for Jetson deployment."
            ) from exc

        return YOLO(str(model_path))
