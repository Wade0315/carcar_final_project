import logging
import os
from pathlib import Path
import time

import cv2
import numpy as np

from camera_base import CameraBase
from performance_logger import PerformanceLogger

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = "/home/waryt/YOLO/best_ncnn_model_v5nu"
FRAME_INTERVAL = 1
DEFAULT_IMAGE_SIZE = 256
DEFAULT_EXPOSURE_TIME_US = 5000


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("logging initialized level=%s", logging.getLevelName(level))


class Camera(CameraBase):
    def __init__(
        self,
        width=320,
        height=240,
        flip_code=-1,
        model_path=DEFAULT_MODEL_PATH,
        confidence=None,
        imgsz=DEFAULT_IMAGE_SIZE,
        target_class=None,
        exposure_time_us=DEFAULT_EXPOSURE_TIME_US,
    ):
        camera_fps = float(os.getenv("YOLO_CAMERA_FPS", "30"))
        exposure_time_us = int(os.getenv("YOLO_EXPOSURE_TIME_US", exposure_time_us))
        super().__init__(
            width,
            height,
            flip_code,
            frame_interval=FRAME_INTERVAL,
            camera_fps=camera_fps,
            exposure_time_us=exposure_time_us,
        )
        default_perf_log = (
            Path(__file__).resolve().parent
            / "logs"
            / f"yolo_performance_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        perf_log_path = os.getenv("YOLO_PERF_LOG", str(default_perf_log)).strip()
        summary_interval = int(os.getenv("YOLO_PERF_SUMMARY_INTERVAL", "30"))
        self.performance_logger = (
            PerformanceLogger(perf_log_path, summary_interval) if perf_log_path else None
        )
        self.last_performance = {}
        self.last_ncnn_performance = {}
        self.last_performance_recorded_at = None

        self.model_path = os.getenv("YOLO_MODEL", model_path)
        self.confidence = float(os.getenv("YOLO_CONF", confidence or 0.25))
        self.imgsz = int(os.getenv("YOLO_IMGSZ", imgsz or max(self.width, self.height)))
        self.target_class = os.getenv("YOLO_CLASS", target_class or "").strip().lower()
        self.iou_threshold = float(os.getenv("YOLO_IOU", "0.45"))
        self.class_names = self.load_class_names(self.model_path)
        self.last_detection_count = 0
        self.last_candidate_count = 0
        self.last_max_confidence = None

        self.open_camera(warmup_seconds=1)
        self.model = self.load_model(self.model_path)
        self.start_frame_capture()
        time.sleep(2)
        logger.info(
            "tracking config camera_fps=%.1f frame_interval=%s camera_frame_period_ms=%.1f imgsz=%s exposure_time_us=%s",
            self.camera_fps,
            self.frame_interval,
            self.camera_frame_period_ms,
            self.imgsz,
            self.exposure_time_us,
        )

    def load_model(self, model_path):
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError(
                "ncnn is required for camera_YOLO.py. Install the NCNN Python runtime on the Raspberry Pi."
            ) from exc

        self.ncnn = ncnn
        model_dir = Path(model_path).expanduser()
        param_path = self.find_model_file(model_dir, ".param")
        bin_path = param_path.with_suffix(".bin")
        if not bin_path.exists():
            raise FileNotFoundError(f"NCNN bin file not found: {bin_path}")

        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        net.opt.num_threads = 4
        net.opt.use_packing_layout = True
        net.opt.use_fp16_storage = True

        logger.info("loading NCNN model param=%s bin=%s", param_path, bin_path)
        if net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"failed to load NCNN param file: {param_path}")
        if net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"failed to load NCNN bin file: {bin_path}")
        return net

    def find_model_file(self, model_path, suffix):
        if model_path.is_file() and model_path.suffix == suffix:
            return model_path
        if not model_path.exists():
            raise FileNotFoundError(f"NCNN model path not found: {model_path}")

        matches = sorted(model_path.glob(f"*{suffix}"))
        if not matches:
            raise FileNotFoundError(f"no {suffix} file found in {model_path}")
        return matches[0]

    def load_class_names(self, model_path):
        names_env = os.getenv("YOLO_NAMES", "").strip()
        if names_env:
            return {idx: name.strip().lower() for idx, name in enumerate(names_env.split(","))}

        metadata_path = Path(model_path).expanduser() / "metadata.yaml"
        if not metadata_path.exists():
            return {}

        try:
            import yaml
        except ImportError:
            logger.warning("metadata.yaml found but PyYAML is not installed; class names will use class ids")
            return {}

        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        names = metadata.get("names", {})
        if isinstance(names, list):
            return {idx: str(name).lower() for idx, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(idx): str(name).lower() for idx, name in names.items()}
        return {}

    def detect_frame(self, frame):
        floor_mask = self.build_floor_mask(frame)
        candidates = self.detect_yolo_candidates(frame)
        find_ball, error, target = self.choose_ball(candidates)
        return floor_mask, candidates, target, find_ball, error

    def detect_yolo_candidates(self, frame):
        self.last_detection_count = 0
        self.last_candidate_count = 0
        self.last_max_confidence = None
        #proprocess inference decode
        started_at = time.perf_counter()
        input_image, scale, pad_x, pad_y = self.preprocess_for_ncnn(frame)
        preprocessed_at = time.perf_counter()
        outputs = self.run_ncnn(input_image)
        inferred_at = time.perf_counter()
        if not outputs:
            self.update_detection_performance(started_at, preprocessed_at, inferred_at)
            logger.warning("NCNN returned no outputs")
            return []

        logger.debug("NCNN output shapes: %s", [output.shape for output in outputs])
        detections = self.decode_yolo_outputs(outputs)
        #detection: {"box": (x1, y1, x2, y2),"confidence": confidence,"class_id": class_id,}
        if not detections:
            self.update_detection_performance(started_at, preprocessed_at, inferred_at)
            logger.info("no detections above confidence %.2f", self.confidence)
            return []
        #postprocess
        detections = self.nms(detections)
        self.last_detection_count = len(detections)
        #find candidate
        candidates = self.build_candidates(detections, scale, pad_x, pad_y)
        candidates = self.group_shuttle_candidates(candidates)
        self.last_candidate_count = len(candidates)
        self.update_detection_performance(started_at, preprocessed_at, inferred_at)
        logger.info("detections=%s candidates=%s", len(detections), len(candidates))
        return candidates

    def update_detection_performance(self, started_at, preprocessed_at, inferred_at):
        finished_at = time.perf_counter()
        self.last_performance = {
            "preprocess_ms": (preprocessed_at - started_at) * 1000,
            "inference_ms": (inferred_at - preprocessed_at) * 1000,
            "postprocess_ms": (finished_at - inferred_at) * 1000,
        }

    def record_performance(self, frame_index, capture_ms, processing_ms, find_ball, error):
        if self.performance_logger is None:
            return

        recorded_at = time.perf_counter()
        processed_gap_ms = None
        effective_fps = None
        if self.last_performance_recorded_at is not None:
            processed_gap_ms = (recorded_at - self.last_performance_recorded_at) * 1000
            effective_fps = 1000 / processed_gap_ms
        self.last_performance_recorded_at = recorded_at

        sample = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_index": frame_index,
            "frame_interval": self.frame_interval,
            "camera_frame_period_ms": round(self.camera_frame_period_ms, 3),
            "capture_ms": round(capture_ms, 3),
            "preprocess_ms": round(self.last_performance.get("preprocess_ms", 0), 3),
            "inference_ms": round(self.last_performance.get("inference_ms", 0), 3),
            "ncnn_prepare_ms": round(self.last_ncnn_performance.get("ncnn_prepare_ms", 0), 3),
            "ncnn_normalize_ms": round(self.last_ncnn_performance.get("ncnn_normalize_ms", 0), 3),
            "ncnn_input_ms": round(self.last_ncnn_performance.get("ncnn_input_ms", 0), 3),
            "ncnn_extract_ms": round(self.last_ncnn_performance.get("ncnn_extract_ms", 0), 3),
            "ncnn_numpy_ms": round(self.last_ncnn_performance.get("ncnn_numpy_ms", 0), 3),
            "postprocess_ms": round(self.last_performance.get("postprocess_ms", 0), 3),
            "processing_ms": round(processing_ms, 3),
            "processed_gap_ms": round(processed_gap_ms, 3) if processed_gap_ms is not None else None,
            "effective_fps": round(effective_fps, 3) if effective_fps is not None else None,
            "detections": self.last_detection_count,
            "candidates": self.last_candidate_count,
            "find_ball": int(find_ball),
            "error": error,
        }
        self.performance_logger.record(sample)

    def preprocess_for_ncnn(self, frame):
        source_h, source_w = frame.shape[:2]
        scale = min(self.imgsz / source_w, self.imgsz / source_h)
        resized_w = int(round(source_w * scale))
        resized_h = int(round(source_h * scale))
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - resized_w) // 2
        pad_y = (self.imgsz - resized_h) // 2
        padded[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized

        return padded, scale, pad_x, pad_y

    def run_ncnn(self, input_image):
        started_at = time.perf_counter()
        input_name = self.model.input_names()[0]
        output_names = sorted(self.model.output_names())
        input_image = np.ascontiguousarray(input_image)
        input_mat = self.ncnn.Mat.from_pixels(
            input_image,
            self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.imgsz,
            self.imgsz,
        )
        prepared_at = time.perf_counter()
        input_mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
        normalized_at = time.perf_counter()

        with self.model.create_extractor() as extractor:
            extractor.input(input_name, input_mat)
            input_at = time.perf_counter()
            outputs = []
            extract_ms = 0
            numpy_ms = 0
            for output_name in output_names:
                extract_started_at = time.perf_counter()
                result = extractor.extract(output_name)
                extracted_at = time.perf_counter()
                extract_ms += (extracted_at - extract_started_at) * 1000
                if isinstance(result, tuple):
                    ret, output = result
                    if ret != 0:
                        logger.warning("NCNN extract failed output=%s ret=%s", output_name, ret)
                        continue
                else:
                    output = result
                outputs.append(np.array(output))
                numpy_ms += (time.perf_counter() - extracted_at) * 1000

        self.last_ncnn_performance = {
            "ncnn_prepare_ms": (prepared_at - started_at) * 1000,
            "ncnn_normalize_ms": (normalized_at - prepared_at) * 1000,
            "ncnn_input_ms": (input_at - normalized_at) * 1000,
            "ncnn_extract_ms": extract_ms,
            "ncnn_numpy_ms": numpy_ms,
        }
        return outputs

    def decode_yolo_outputs(self, outputs):
        detections = []
        max_confidence = None
        for output in outputs:
            predictions = self.normalize_output_shape(output)
            if predictions is None:
                continue

            logger.debug("NCNN prediction shape=%s", predictions.shape)
            output_detections, output_max_confidence = self.decode_predictions(predictions)
            detections.extend(output_detections)
            if output_max_confidence is not None:
                if max_confidence is None or output_max_confidence > max_confidence:
                    max_confidence = output_max_confidence

        if max_confidence is not None:
            self.last_max_confidence = max_confidence
            logger.info("max raw confidence=%.4f threshold=%.4f", max_confidence, self.confidence)
        return detections

    def decode_predictions(self, predictions):
        if predictions.shape[1] == 6 and self.is_postprocessed_output(predictions):
            return self.decode_postprocessed_predictions(predictions)
        return self.decode_raw_predictions(predictions)

    def is_postprocessed_output(self, predictions):
        class_count = max(len(self.class_names), 1)
        class_ids = predictions[:, 5]
        return bool(np.all(
            (predictions[:, 4] >= 0)
            & (predictions[:, 4] <= 1)
            & (np.abs(class_ids - np.rint(class_ids)) <= 1e-3)
            & (predictions[:, 2] > predictions[:, 0])
            & (predictions[:, 3] > predictions[:, 1])
            & (class_ids >= 0)
            & (class_ids < class_count)
        ))

    def decode_postprocessed_predictions(self, predictions):
        confidences = predictions[:, 4]
        max_confidence = float(np.max(confidences)) if len(confidences) else None
        selected = predictions[confidences >= self.confidence]
        detections = [
            {
                "box": tuple(float(value) for value in prediction[:4]),
                "confidence": float(prediction[4]),
                "class_id": int(prediction[5]),
            }
            for prediction in selected
        ]
        return detections, max_confidence

    def decode_raw_predictions(self, predictions):
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        max_confidence = float(np.max(confidences)) if len(confidences) else None
        selected_indexes = np.flatnonzero(confidences >= self.confidence)

        detections = []
        for index in selected_indexes:
            cx, cy, w, h = (float(value) for value in predictions[index, :4])
            detections.append({
                "box": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                "confidence": float(confidences[index]),
                "class_id": int(class_ids[index]),
            })
        return detections, max_confidence

    def normalize_output_shape(self, output):
        output = np.asarray(output)
        output = np.squeeze(output)
        if output.ndim != 2:
            logger.warning("unsupported NCNN output shape: %s", output.shape)
            return None

        if output.shape[0] < output.shape[1] and output.shape[0] <= 256:
            output = output.T

        if output.shape[1] < 5:
            logger.warning("unsupported NCNN prediction shape: %s", output.shape)
            return None
        return output

    def nms(self, detections):
        detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        kept = []

        while detections:
            best = detections.pop(0)
            kept.append(best)
            detections = [
                item for item in detections
                if item["class_id"] != best["class_id"] or self.iou(item["box"], best["box"]) < self.iou_threshold
            ]

        return kept

    def required_confidence(self, area):
        if area < 300:
            return 0.30
        if area < 1000:
            return 0.45
        if area < 3000:
            return 0.55
        return 0.75

    def iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0
        return intersection / union

    def build_candidates(self, detections, scale, pad_x, pad_y):
        candidates = []
        for detection in detections:
            class_id = detection["class_id"]
            class_name = str(self.class_names.get(class_id, class_id)).lower()
            if self.target_class and self.target_class not in class_name:
                continue

            x1, y1, x2, y2 = detection["box"]
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = int(max(0, min(self.width - 1, round(x1))))
            y1 = int(max(0, min(self.height - 1, round(y1))))
            x2 = int(max(0, min(self.width - 1, round(x2))))
            y2 = int(max(0, min(self.height - 1, round(y2))))
            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1
            target_cx = x1 + w // 2
            target_cy = y1 + h // 2
            error_from_center = target_cx - self.width // 2
            ratio = min(w, h) / max(w, h)
            area = w * h
            min_confidence = self.required_confidence(area)
            if detection["confidence"] < min_confidence:
                logger.debug(
                    "reject candidate area=%s confidence=%.3f required=%.3f",
                    area,
                    detection["confidence"],
                    min_confidence,
                )
                continue

            candidates.append({
                "rect": ((target_cx, target_cy), (w, h), 0),
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "confidence": detection["confidence"],
                "required_confidence": min_confidence,
                "class_id": class_id,
                "class_name": class_name,
                "is_head": "head" in class_name,
                "rect_cx": target_cx,
                "rect_cy": target_cy,
                "target_cx": target_cx,
                "target_cy": target_cy,
                "error": error_from_center,
                "w_h_ratio": ratio,
                "head_found": True,
                "source": "yolo",
            })

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return candidates

    def group_shuttle_candidates(self, candidates):
        heads = [candidate for candidate in candidates if candidate.get("is_head")]
        wholes = [candidate for candidate in candidates if not candidate.get("is_head")]
        if not heads or not wholes:
            return candidates

        grouped = []
        used_heads = set()
        used_wholes = set()

        for whole_index, whole in enumerate(wholes):
            matches = []
            for head_index, head in enumerate(heads):
                if head_index in used_heads:
                    continue
                if self.head_matches_whole(head, whole):
                    matches.append((head_index, head))

            if not matches:
                continue

            head_index, head = max(matches, key=lambda item: item[1]["confidence"])
            grouped.append(self.merge_shuttle_candidate(head, whole))
            used_heads.add(head_index)
            used_wholes.add(whole_index)

        for head_index, head in enumerate(heads):
            if head_index not in used_heads:
                grouped.append(head)
        for whole_index, whole in enumerate(wholes):
            if whole_index not in used_wholes:
                grouped.append(whole)

        grouped.sort(key=lambda item: item["confidence"], reverse=True)
        return grouped

    def head_matches_whole(self, head, whole):
        return (
            self.point_in_bbox((head["target_cx"], head["target_cy"]), whole["bbox"])
            or self.bboxes_intersect(head["bbox"], whole["bbox"])
        )

    def point_in_bbox(self, point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def bboxes_intersect(self, bbox_a, bbox_b):
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        return min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)

    def merge_shuttle_candidate(self, head, whole):
        x1 = min(head["bbox"][0], whole["bbox"][0])
        y1 = min(head["bbox"][1], whole["bbox"][1])
        x2 = max(head["bbox"][2], whole["bbox"][2])
        y2 = max(head["bbox"][3], whole["bbox"][3])
        w = x2 - x1
        h = y2 - y1
        area = w * h
        target_cx = head["target_cx"]
        target_cy = head["target_cy"]

        merged = dict(head)
        merged.update({
            "rect": ((target_cx, target_cy), (w, h), 0),
            "bbox": (x1, y1, x2, y2),
            "area": area,
            "confidence": max(head["confidence"], whole["confidence"]),
            "required_confidence": max(
                head.get("required_confidence", 0),
                whole.get("required_confidence", 0),
            ),
            "class_name": f"{head['class_name']}+{whole['class_name']}",
            "is_head": True,
            "rect_cx": target_cx,
            "rect_cy": target_cy,
            "target_cx": target_cx,
            "target_cy": target_cy,
            "error": target_cx - self.width // 2,
            "w_h_ratio": min(w, h) / max(w, h),
            "head_found": True,
            "source": "yolo_grouped",
            "head_bbox": head["bbox"],
            "whole_bbox": whole["bbox"],
            "head_confidence": head["confidence"],
            "whole_confidence": whole["confidence"],
        })
        return merged

    def process_frame(self, frame):
        _, _, target, find_ball, error = self.detect_frame(frame)
        return find_ball, error, target

    def streaming(self):
        logger.info("Starting YOLO tracking...")
        last_frame_index = None

        try:
            while True:
                raw_frame, capture_ms, frame_index = self.get_latest_frame(last_frame_index)
                last_frame_index = frame_index
                processing_started_at = time.perf_counter()
                find_ball, error, target = self.process_frame(raw_frame)
                processing_ms = (time.perf_counter() - processing_started_at) * 1000
                self.record_performance(frame_index, capture_ms, processing_ms, find_ball, error)
                yield find_ball, error, target

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def close(self):
        already_closed = self.closed
        super().close()
        performance_logger = getattr(self, "performance_logger", None)
        if not already_closed and performance_logger is not None:
            performance_logger.close()


if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        # tracker.single_test()
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
