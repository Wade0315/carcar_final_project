import csv
import logging
from collections import deque
from pathlib import Path


logger = logging.getLogger(__name__)
PERF_LOG_FIELDS = [
    "timestamp",
    "frame_index",
    "frame_interval",
    "camera_frame_period_ms",
    "capture_ms",
    "preprocess_ms",
    "inference_ms",
    "ncnn_prepare_ms",
    "ncnn_normalize_ms",
    "ncnn_input_ms",
    "ncnn_extract_ms",
    "ncnn_numpy_ms",
    "postprocess_ms",
    "processing_ms",
    "processed_gap_ms",
    "effective_fps",
    "detections",
    "candidates",
    "find_ball",
    "error",
]


class PerformanceLogger:
    def __init__(self, path, summary_interval=30):
        self.path = Path(path).expanduser().resolve()
        self.summary_interval = max(1, summary_interval)
        self.samples = deque(maxlen=self.summary_interval)
        self.total_samples = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=PERF_LOG_FIELDS)
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()
        logger.info("performance log=%s summary_interval=%s", self.path, self.summary_interval)

    def record(self, sample):
        self.writer.writerow({field: sample.get(field) for field in PERF_LOG_FIELDS})
        self.file.flush()
        self.samples.append(sample)
        self.total_samples += 1
        if self.total_samples == 1:
            logger.info("performance first sample written path=%s", self.path)

        if self.total_samples % self.summary_interval == 0:
            self.log_summary()

    def log_summary(self):
        if not self.samples:
            return

        inference_times = [sample["inference_ms"] for sample in self.samples]
        processing_times = [sample["processing_ms"] for sample in self.samples]
        effective_fps_values = [
            sample["effective_fps"] for sample in self.samples if sample["effective_fps"] is not None
        ]
        logger.info(
            "perf summary samples=%s inference_ms(avg=%.1f max=%.1f) "
            "processing_ms(avg=%.1f max=%.1f) effective_fps(avg=%.1f)",
            len(self.samples),
            sum(inference_times) / len(inference_times),
            max(inference_times),
            sum(processing_times) / len(processing_times),
            max(processing_times),
            sum(effective_fps_values) / len(effective_fps_values) if effective_fps_values else 0,
        )

    def close(self):
        self.log_summary()
        self.file.flush()
        self.file.close()
        logger.info("performance log closed path=%s samples=%s", self.path, self.total_samples)
