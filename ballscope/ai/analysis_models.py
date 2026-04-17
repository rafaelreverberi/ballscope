from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


BALL_LABEL_HINTS = {
    "ball",
    "sports ball",
    "soccer ball",
    "soccerball",
    "football",
    "football ball",
}


@dataclass(frozen=True)
class AnalysisDetection:
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return float(max(1, self.x2 - self.x1) * max(1, self.y2 - self.y1))


@dataclass(frozen=True)
class AnalysisModelMetadata:
    path: str
    backend: str
    label: str
    class_names: tuple[str, ...]
    default_class_id: Optional[int]
    resolution: Optional[int]
    num_classes: Optional[int]
    supports_iou: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "backend": self.backend,
            "label": self.label,
            "class_names": list(self.class_names),
            "default_class_id": self.default_class_id,
            "resolution": self.resolution,
            "num_classes": self.num_classes,
            "supports_iou": self.supports_iou,
        }


def _normalize_label(value: Any) -> str:
    return str(value or "").strip()


def _default_class_id_from_names(class_names: Sequence[str]) -> Optional[int]:
    for idx, name in enumerate(class_names):
        if _normalize_label(name).lower() in BALL_LABEL_HINTS:
            return idx
    return 0 if class_names else None


def _safe_checkpoint_metadata(path: str) -> Optional[AnalysisModelMetadata]:
    if torch is None:
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if not isinstance(checkpoint, dict):
        return None

    args = checkpoint.get("args")
    model_state = checkpoint.get("model")
    if args is not None and isinstance(model_state, dict):
        class_names = list(getattr(args, "class_names", None) or [])
        resolution = getattr(args, "resolution", None)
        num_classes = getattr(args, "num_classes", None)
        if class_names or resolution or "class_embed.bias" in model_state:
            return AnalysisModelMetadata(
                path=path,
                backend="rfdetr",
                label="RF-DETR",
                class_names=tuple(_normalize_label(name) for name in class_names),
                default_class_id=_default_class_id_from_names(class_names),
                resolution=int(resolution) if resolution else None,
                num_classes=int(num_classes) if num_classes else (len(class_names) or None),
                supports_iou=False,
            )
    return None


def inspect_model_metadata(model_path: str) -> AnalysisModelMetadata:
    path = str(Path(model_path).as_posix())
    suffix = Path(path).suffix.lower()

    rf_meta = _safe_checkpoint_metadata(path)
    if rf_meta is not None:
        return rf_meta

    if suffix in {".pt", ".onnx", ".engine"}:
        try:
            from ultralytics import YOLO

            model = YOLO(path)
            names = getattr(model, "names", None)
            if names is None and hasattr(model, "model"):
                names = getattr(model.model, "names", None)
            class_names: List[str] = []
            if isinstance(names, dict):
                for key in sorted(names):
                    class_names.append(_normalize_label(names[key]))
            elif isinstance(names, list):
                class_names = [_normalize_label(name) for name in names]
            return AnalysisModelMetadata(
                path=path,
                backend="yolo",
                label="YOLO",
                class_names=tuple(class_names),
                default_class_id=_default_class_id_from_names(class_names),
                resolution=None,
                num_classes=len(class_names) or None,
                supports_iou=True,
            )
        except Exception:
            pass

    return AnalysisModelMetadata(
        path=path,
        backend="unknown",
        label="Unknown",
        class_names=tuple(),
        default_class_id=None,
        resolution=None,
        num_classes=None,
        supports_iou=False,
    )


class BaseAnalysisModel:
    def __init__(self, metadata: AnalysisModelMetadata):
        self.metadata = metadata

    @property
    def class_names(self) -> List[str]:
        return list(self.metadata.class_names)

    def predict(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        class_id: Optional[int],
        device: str,
    ) -> List[AnalysisDetection]:
        raise NotImplementedError


class YoloAnalysisModel(BaseAnalysisModel):
    def __init__(self, metadata: AnalysisModelMetadata):
        super().__init__(metadata)
        from ultralytics import YOLO

        self.model = YOLO(metadata.path)
        try:
            self.model.fuse()
        except Exception:
            pass

    def predict(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        class_id: Optional[int],
        device: str,
    ) -> List[AnalysisDetection]:
        kwargs: Dict[str, Any] = {
            "imgsz": max(224, int(imgsz)),
            "conf": float(conf),
            "iou": float(iou),
            "device": device,
            "verbose": False,
        }
        if class_id is not None:
            kwargs["classes"] = [int(class_id)]
        results = self.model.predict(image, **kwargs)
        detections: List[AnalysisDetection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections
        names = getattr(results[0], "names", None) or getattr(self.model, "names", {}) or {}
        for box in boxes:
            try:
                det_class_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = _normalize_label(names.get(det_class_id, ""))
                detections.append(
                    AnalysisDetection(
                        class_id=det_class_id,
                        confidence=score,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        label=label,
                    )
                )
            except Exception:
                continue
        return detections


class RFDetrAnalysisModel(BaseAnalysisModel):
    def __init__(self, metadata: AnalysisModelMetadata, device: str):
        super().__init__(metadata)
        try:
            from rfdetr import RFDETRLarge
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"RF-DETR support is not installed: {exc}") from exc

        kwargs: Dict[str, Any] = {
            "pretrain_weights": metadata.path,
            "device": device,
        }
        if metadata.num_classes:
            kwargs["num_classes"] = int(metadata.num_classes)
        if metadata.resolution:
            kwargs["resolution"] = int(metadata.resolution)
        self.model = RFDETRLarge(**kwargs)

    def predict(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        imgsz: int,
        class_id: Optional[int],
        device: str,
    ) -> List[AnalysisDetection]:
        del iou, device
        infer_shape = max(256, min(1408, int(imgsz)))
        detections = self.model.predict(image, threshold=float(conf), shape=(infer_shape, infer_shape))
        xyxy = getattr(detections, "xyxy", None)
        class_ids = getattr(detections, "class_id", None)
        confidences = getattr(detections, "confidence", None)
        labels = getattr(getattr(detections, "data", None), "get", lambda _k, _d=None: None)("class_name", None)
        out: List[AnalysisDetection] = []
        if xyxy is None or class_ids is None or confidences is None:
            return out
        names = list(self.metadata.class_names)
        for idx in range(len(xyxy)):
            det_class_id = int(class_ids[idx])
            if class_id is not None and det_class_id != int(class_id):
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy[idx]]
            label = ""
            if labels is not None and len(labels) > idx:
                label = _normalize_label(labels[idx])
            elif 0 <= det_class_id < len(names):
                label = _normalize_label(names[det_class_id])
            out.append(
                AnalysisDetection(
                    class_id=det_class_id,
                    confidence=float(confidences[idx]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=label,
                )
            )
        return out


def build_analysis_model(metadata: AnalysisModelMetadata, device: str) -> BaseAnalysisModel:
    if metadata.backend == "yolo":
        return YoloAnalysisModel(metadata)
    if metadata.backend == "rfdetr":
        return RFDetrAnalysisModel(metadata, device=device)
    raise RuntimeError(f"Unsupported analysis model format: {metadata.path}")


def metadata_from_paths(paths: Sequence[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in paths:
        meta = inspect_model_metadata(path)
        row = meta.to_dict()
        row["path"] = path
        items.append(row)
    return items
