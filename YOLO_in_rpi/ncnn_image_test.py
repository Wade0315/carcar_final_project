import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from camera_base import CameraBase
from camera_YOLO import Camera as YOLOCamera
from camera_YOLO import setup_logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] /"YOLO"/ "runs" / "detect" / "shuttle_yolov8n_320" /"weights" / "best_ncnn_model"
DEFAULT_STOCK_DIR = Path(__file__).resolve().parents[1] / "YOLO" / "stock"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ncnn_test"


class NcnnImageTester(YOLOCamera):
    def __init__(
        self,
        width=320,
        height=240,
        flip_code=None,
        model_path=DEFAULT_MODEL_PATH,
        confidence=None,
        imgsz=None,
        target_class=None,
    ):
        CameraBase.__init__(self, width, height, flip_code)
        self.model_path = os.getenv("YOLO_MODEL", model_path)
        self.confidence = float(os.getenv("YOLO_CONF", confidence or 0.25))
        self.imgsz = int(os.getenv("YOLO_IMGSZ", imgsz or max(self.width, self.height)))
        self.target_class = os.getenv("YOLO_CLASS", target_class or "").strip().lower()
        self.iou_threshold = float(os.getenv("YOLO_IOU", "0.45"))
        self.class_names = self.load_class_names(self.model_path)
        self.model = self.load_model(self.model_path)

    def draw_candidates(self, frame, candidates):
        annotated = frame.copy()
        cv2.line(annotated, (self.width // 2, 0), (self.width // 2, self.height), (0, 255, 255), 1)

        for candidate in candidates:
            x1, y1, x2, y2 = candidate["bbox"]
            target_cx = candidate["target_cx"]
            target_cy = candidate["target_cy"]
            label = f"{candidate['class_name']} {candidate['confidence']:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(annotated, (target_cx, target_cy), 4, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
            )
            cv2.putText(
                annotated,
                f"error={candidate['error']}",
                (x1, min(y2 + 15, self.height - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        return annotated

    def inspect_outputs(self, frame):
        input_image, _, _, _ = self.preprocess_for_ncnn(frame)
        outputs = self.run_ncnn(input_image)
        logger.info("input image shape=%s min=%s max=%s", input_image.shape, input_image.min(), input_image.max())
        logger.info("output count=%s", len(outputs))

        for index, output in enumerate(outputs):
            array = np.asarray(output)
            squeezed = np.squeeze(array)
            logger.info(
                "output[%s] shape=%s squeezed=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                index,
                array.shape,
                squeezed.shape,
                array.dtype,
                float(np.min(array)),
                float(np.max(array)),
                float(np.mean(array)),
            )
            logger.info("output[%s] first values=%s", index, np.ravel(squeezed)[:24])
            self.inspect_layout_candidates(squeezed, index)

    def inspect_layout_candidates(self, output, output_index):
        if output.ndim != 2:
            return

        layouts = {
            "as_is": output,
            "transposed": output.T,
        }
        for name, predictions in layouts.items():
            if predictions.ndim != 2 or predictions.shape[1] < 5:
                continue

            class_scores = predictions[:, 4:]
            top_scores = np.max(class_scores, axis=1)
            top_indices = np.argsort(top_scores)[-5:][::-1]
            logger.info(
                "output[%s] layout=%s predictions=%s attrs=%s top_scores=%s",
                output_index,
                name,
                predictions.shape[0],
                predictions.shape[1],
                top_scores[top_indices],
            )

            for rank, pred_index in enumerate(top_indices, start=1):
                prediction = predictions[pred_index]
                logger.info(
                    "output[%s] layout=%s top%s index=%s box=%s scores=%s",
                    output_index,
                    name,
                    rank,
                    pred_index,
                    prediction[:4],
                    prediction[4:12],
                )


def list_images(source):
    source = Path(source).expanduser()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"unsupported image file: {source}")
        return [source]

    if not source.exists():
        raise FileNotFoundError(f"image source not found: {source}")

    images = [
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda path: path.name)
    if not images:
        raise FileNotFoundError(f"no images found in {source}")
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Run NCNN YOLO on stock images and draw candidate boxes.")
    parser.add_argument("--source", default=os.getenv("STOCK_IMAGE_PATH", str(DEFAULT_STOCK_DIR)))
    parser.add_argument("--output", default=os.getenv("NCNN_IMAGE_TEST_OUTPUT", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", DEFAULT_MODEL_PATH))
    parser.add_argument("--width", type=int, default=int(os.getenv("CAMERA_WIDTH", "320")))
    parser.add_argument("--height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "240")))
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--target-class", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    tester = NcnnImageTester(
        width=args.width,
        height=args.height,
        model_path=args.model,
        confidence=args.conf,
        imgsz=args.imgsz,
        target_class=args.target_class,
    )

    images = list_images(args.source)
    if args.limit is not None:
        images = images[:args.limit]

    for index, image_path in enumerate(images, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning("failed to read %s", image_path)
            continue

        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        frame = tester.fix_orientation(frame)

        if args.inspect:
            logger.info("inspecting %s", image_path)
            tester.inspect_outputs(frame)
            break

        candidates = tester.detect_yolo_candidates(frame)
        annotated = tester.draw_candidates(frame, candidates)

        output_path = output_dir / f"{image_path.stem}_candidates.jpg"
        cv2.imwrite(str(output_path), annotated)
        logger.info(
            "[%s/%s] %s candidates=%s -> %s",
            index,
            len(images),
            image_path.name,
            len(candidates),
            output_path,
        )

        if args.show:
            cv2.imshow("NCNN Candidates", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
