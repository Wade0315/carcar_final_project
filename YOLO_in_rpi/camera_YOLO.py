import logging
import os
from pathlib import Path
import time

import cv2
import numpy as np

from camera_base import CameraBase

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = "/home/waryt/YOLO/best_ncnn_model"


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
        imgsz=None,
        target_class=None,
    ):
        super().__init__(width, height, flip_code)

        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is required for camera_YOLO.py. Use cameraFAKE.py for local image preview."
            ) from exc

        self.picam2 = Picamera2()
        self.closed = False
        self.model_path = os.getenv("YOLO_MODEL", model_path)
        self.confidence = float(os.getenv("YOLO_CONF", confidence or 0.25))
        self.imgsz = int(os.getenv("YOLO_IMGSZ", imgsz or max(self.width, self.height)))
        self.target_class = os.getenv("YOLO_CLASS", target_class or "").strip().lower()
        self.iou_threshold = float(os.getenv("YOLO_IOU", "0.45"))
        self.class_names = self.load_class_names(self.model_path)
        self.model = self.load_model(self.model_path)

        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (self.width, self.height)},
            controls={"FrameRate": 30},
            buffer_count=3,
        )
        self.picam2.configure(config)
        self.picam2.start()

        logger.info("camera activating...")
        time.sleep(2)
        self.lock_current_camera_controls()

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

    def lock_current_camera_controls(self):
        metadata = self.picam2.capture_metadata()
        exposure_time = metadata.get("ExposureTime")
        analogue_gain = metadata.get("AnalogueGain")
        colour_gains = metadata.get("ColourGains")

        controls = {
            "AeEnable": False,
            "AwbEnable": False,
        }
        if exposure_time is not None:
            controls["ExposureTime"] = exposure_time
        if analogue_gain is not None:
            controls["AnalogueGain"] = analogue_gain
        if colour_gains is not None:
            controls["ColourGains"] = colour_gains

        self.picam2.set_controls(controls)
        logger.info(
            "lock camera ExposureTime=%s AnalogueGain=%s ColourGains=%s",
            exposure_time,
            analogue_gain,
            colour_gains,
        )

    def detect_frame(self, frame):
        floor_mask = self.build_floor_mask(frame)
        candidates = self.detect_yolo_candidates(frame)
        find_ball, error, target = self.choose_ball(candidates)
        return floor_mask, candidates, target, find_ball, error

    def detect_yolo_candidates(self, frame):
        input_image, scale, pad_x, pad_y = self.preprocess_for_ncnn(frame)
        outputs = self.run_ncnn(input_image)
        if not outputs:
            logger.warning("NCNN returned no outputs")
            return []

        logger.debug("NCNN output shapes: %s", [output.shape for output in outputs])
        detections = self.decode_yolo_outputs(outputs)
        if not detections:
            logger.info("no detections above confidence %.2f", self.confidence)
            return []

        detections = self.nms(detections)
        candidates = self.build_candidates(detections, scale, pad_x, pad_y)
        logger.info("detections=%s candidates=%s", len(detections), len(candidates))
        return candidates

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
        input_name = self.model.input_names()[0]
        output_names = sorted(self.model.output_names())
        input_image = np.ascontiguousarray(input_image)
        input_mat = self.ncnn.Mat.from_pixels(
            input_image,
            self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.imgsz,
            self.imgsz,
        )
        input_mat.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        with self.model.create_extractor() as extractor:
            extractor.input(input_name, input_mat)
            outputs = []
            for output_name in output_names:
                result = extractor.extract(output_name)
                if isinstance(result, tuple):
                    ret, output = result
                    if ret != 0:
                        logger.warning("NCNN extract failed output=%s ret=%s", output_name, ret)
                        continue
                else:
                    output = result
                outputs.append(np.array(output))
        return outputs

    def decode_yolo_outputs(self, outputs):
        detections = []
        max_confidence = None
        for output in outputs:
            predictions = self.normalize_output_shape(output)
            if predictions is None:
                continue

            logger.debug("NCNN prediction shape=%s", predictions.shape)
            for prediction in predictions:
                if len(prediction) < 5:
                    continue

                detection = self.decode_prediction(prediction)
                if detection is None:
                    continue

                confidence = detection["confidence"]
                if max_confidence is None or confidence > max_confidence:
                    max_confidence = confidence
                if confidence < self.confidence:
                    continue

                detections.append(detection)

        if max_confidence is not None:
            logger.info("max raw confidence=%.4f threshold=%.4f", max_confidence, self.confidence)
        return detections

    def decode_prediction(self, prediction):
        if self.is_postprocessed_detection(prediction):
            x1, y1, x2, y2, confidence, class_id = [float(value) for value in prediction]
            return {
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": int(class_id),
            }

        box = prediction[:4]
        scores = prediction[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        cx, cy, w, h = [float(value) for value in box]
        return {
            "box": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
            "confidence": confidence,
            "class_id": class_id,
        }

    def is_postprocessed_detection(self, prediction):
        if len(prediction) != 6:
            return False

        x1, y1, x2, y2, confidence, class_id = [float(value) for value in prediction]
        if not (0 <= confidence <= 1):
            return False
        if abs(class_id - round(class_id)) > 1e-3:
            return False
        if x2 <= x1 or y2 <= y1:
            return False

        class_count = max(len(self.class_names), 1)
        return 0 <= int(round(class_id)) < class_count

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

            candidates.append({
                "rect": ((target_cx, target_cy), (w, h), 0),
                "bbox": (x1, y1, x2, y2),
                "area": w * h,
                "confidence": detection["confidence"],
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

    def process_frame(self, frame):
        _, _, _, find_ball, error = self.detect_frame(frame)
        return find_ball, error

    def streaming(self):
        logger.info("Starting YOLO tracking...")
        at_frame = 0

        try:
            while True:
                raw_frame = self.picam2.capture_array()
                raw_frame = self.fix_orientation(raw_frame)

                if at_frame % 3 == 0:
                    find_ball, error = self.process_frame(raw_frame)
                    yield find_ball, error

                at_frame += 1

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        finally:
            self.close()

    def close(self):
        if self.closed:
            return
        self.picam2.stop()
        self.closed = True
        logger.info("Camera closed.")


if __name__ == "__main__":
    setup_logging()
    with Camera() as tracker:
        # tracker.single_test()
        for find_ball, error in tracker.streaming():
            logger.info("find_ball=%s error=%s", find_ball, error)
