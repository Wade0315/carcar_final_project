import os
import logging
import time
from enum import Enum
import arduino
import camera_YOLO as camera
#import cameraUI as camera

class Status(Enum):
    TRACK = 0
    NOT_FOUND = 1
    CLOSE_ENOUGH = 2
    OUT_OF_BOUND = 3
    IDLE = 4

FOUND_TOLERANCE = 2         
CLOSE_TRACK = 20
WARMUP_SECONDS = float(os.getenv("YOLO_WARMUP_SECONDS", "2"))
WARMUP_STABLE_FRAMES = int(os.getenv("YOLO_WARMUP_STABLE_FRAMES", "5"))
MAX_INFERENCE_MS = float(os.getenv("YOLO_MAX_INFERENCE_MS", "800"))
SLOW_INFERENCE_TOLERANCE = max(1, int(os.getenv("YOLO_SLOW_INFERENCE_TOLERANCE", "3")))

logger = logging.getLogger(__name__)


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("logging initialized level=%s", logging.getLevelName(level))

def has_target(target):
    return True if target is not None else False

def is_close_enough_target(target):
    return bool(target and target.get("close_enough"))

def describe_target(target):
    if not has_target(target):
        return ""

    return (
        " source=%s area=%s grouped_area=%s grouped_cx=%s grouped_cy=%s"
        " ball_area=%s ball_cx=%s ball_cy=%s target_cx=%s target_cy=%s close_enough=%s"
        % (
            target.get("source"),
            target.get("area"),
            target.get("grouped_area"),
            target.get("grouped_cx"),
            target.get("grouped_cy"),
            target.get("ball_area"),
            target.get("ball_cx"),
            target.get("ball_cy"),
            target.get("target_cx"),
            target.get("target_cy"),
            target.get("close_enough"),
        )
    )

def main():

    found_count = 0
    last_sent_state = None

    try:
        with camera.Camera() as cam:
            state = Status.NOT_FOUND
            mega = arduino.Arduino()
            #mega.send(state.value)
            last_sent_state = state
            warmup_ends_at = time.monotonic() + WARMUP_SECONDS
            stable_inference_count = 0
            slow_inference_count = 0
            controls_enabled = False
            logger.info(
                "warming up YOLO for at least %.1f seconds; waiting for %s consecutive "
                "inferences <= %.1f ms before enabling motors; motors stop after %s "
                "consecutive slow inferences",
                WARMUP_SECONDS,
                WARMUP_STABLE_FRAMES,
                MAX_INFERENCE_MS,
                SLOW_INFERENCE_TOLERANCE,
            )

            for ball_detected, error, target in cam.streaming():
                mega.receive()
                inference_ms = cam.last_performance.get("inference_ms", float("inf"))
                if inference_ms <= MAX_INFERENCE_MS:
                    stable_inference_count += 1
                    slow_inference_count = 0
                else:
                    stable_inference_count = 0
                    slow_inference_count += 1
                    if controls_enabled and slow_inference_count < SLOW_INFERENCE_TOLERANCE:
                        logger.warning(
                            "slow inference %.1f ms > %.1f ms; keep motors enabled "
                            "slow_count=%s/%s",
                            inference_ms,
                            MAX_INFERENCE_MS,
                            slow_inference_count,
                            SLOW_INFERENCE_TOLERANCE,
                        )
                    else:
                        logger.warning(
                            "slow inference %.1f ms > %.1f ms; motors remain stopped "
                            "slow_count=%s/%s",
                            inference_ms,
                            MAX_INFERENCE_MS,
                            slow_inference_count,
                            SLOW_INFERENCE_TOLERANCE,
                        )

                inference_ready = stable_inference_count >= WARMUP_STABLE_FRAMES
                if controls_enabled:
                    controls_ready = slow_inference_count < SLOW_INFERENCE_TOLERANCE
                else:
                    controls_ready = time.monotonic() >= warmup_ends_at and inference_ready
                if not controls_ready:
                    found_count = 0
                    state = Status.NOT_FOUND
                    if last_sent_state != state:
                        mega.send(state.value)
                        last_sent_state = state
                    controls_enabled = False
                    continue

                if not controls_enabled:
                    logger.info("YOLO inference stabilized; motors enabled")
                    controls_enabled = True

                if ball_detected:
                    found_count += 1

                    if found_count < FOUND_TOLERANCE:
                        continue

                    if error is None:
                        state = Status.IDLE
                    elif abs(error) <= CLOSE_TRACK:
                        if is_close_enough_target(target) :
                            state = Status.CLOSE_ENOUGH
                            logger.info("%s error=%s%s", state.name, error, describe_target(target))
                        else:
                            state = Status.TRACK
                            logger.info("%s error=%s%s", state.name, error, describe_target(target))
                    else:
                        state = Status.TRACK

                    if state == Status.TRACK:
                        mega.send(f"{state.value} {error}")
                        last_sent_state = state
                        logger.info("%s error=%s%s", state.name, error, describe_target(target))
                    else:
                        mega.send(state.value)
                        if last_sent_state != state:
                            last_sent_state = state
                            logger.info("%s", state.name)

                else:
                    found_count = 0
                    state = Status.NOT_FOUND
                    mega.send(state.value)
                    if last_sent_state != state:
                        last_sent_state = state
                        logger.info("%s", state.name)
    finally:
        mega.close()

if __name__ == "__main__":
    setup_logging()
    main()
