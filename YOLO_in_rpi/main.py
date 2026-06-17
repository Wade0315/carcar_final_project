import os
import logging
import select
import sys
import termios
import time
import tty
from enum import Enum
from pathlib import Path
import arduino
import camera_YOLO as camera
#import cameraUI as camera

class Status(Enum):
    TRACK = 0
    NOT_FOUND = 1
    CLOSE_ENOUGH = 2
    OUT_OF_BOUND = 3
    INIT = 4
    NOHEAD = 5
    IDLE = 6

FOUND_TOLERANCE = 2         
CLOSE_TRACK = 22
ARM_CATCH_TIME = 20
NOHEAD_AREA = int(os.getenv("YOLO_NOHEAD_AREA", "3000"))
NOHEAD_MAX_AREA = int(os.getenv("YOLO_NOHEAD_MAX_AREA", "30000"))
NOHEAD_SLEEP_TIME = float(os.getenv("YOLO_NOHEAD_SLEEP_TIME", "1.0"))
NOHEAD_TOLERANCE = max(1, int(os.getenv("YOLO_NOHEAD_TOLERANCE", "3")))
WARMUP_SECONDS = float(os.getenv("YOLO_WARMUP_SECONDS", "2"))
WARMUP_STABLE_FRAMES = int(os.getenv("YOLO_WARMUP_STABLE_FRAMES", "5"))
MAX_INFERENCE_MS = float(os.getenv("YOLO_MAX_INFERENCE_MS", "800"))
SLOW_INFERENCE_TOLERANCE = max(1, int(os.getenv("YOLO_SLOW_INFERENCE_TOLERANCE", "3")))

logger = logging.getLogger(__name__)


def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    third_party_level_name = os.getenv("THIRD_PARTY_LOG_LEVEL", "WARNING").upper()
    third_party_level = getattr(logging, third_party_level_name, logging.WARNING)
    default_log_path = (Path(__file__).resolve().parent / "logs" / f"system_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_path = Path(os.getenv("SYSTEM_LOG", default_log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    for logger_name in ("picamera2", "picamera2.picamera2", "libcamera"):
        logging.getLogger(logger_name).setLevel(third_party_level)
    logger.info(
        "logging initialized level=%s third_party_level=%s file=%s",
        logging.getLevelName(level),
        logging.getLevelName(third_party_level),
        log_path,
    )

def has_target(target):
    return True if target is not None else False

def is_close_enough_target(target):
    return bool(target and target.get("close_enough"))

def is_close_nohead_target(target):
    if not target or target.get("is_head"):
        return False

    area = target.get("ball_area") or target.get("area") or 0
    return NOHEAD_MAX_AREA >= area >= NOHEAD_AREA

def describe_target(target):
    if not has_target(target):
        return ""

    return (
        " tier=%s selection_mode=%s selection_area=%s selection_distance_sq=%s"
        " source=%s is_head=%s area=%s grouped_area=%s grouped_cx=%s grouped_cy=%s"
        " ball_area=%s ball_cx=%s ball_cy=%s target_cx=%s target_cy=%s close_enough=%s"
        % (
            target.get("tracking_tier"),
            target.get("selection_mode"),
            target.get("selection_area"),
            target.get("selection_distance_sq"),
            target.get("source"),
            target.get("is_head"),
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

class QuitKeyWatcher:
    def __init__(self, quit_key="q"):
        self.quit_key = quit_key
        self.enabled = False
        self.old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.enabled = True
        else:
            logger.warning("stdin is not a terminal; press Ctrl+C to stop")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def pressed(self):
        if not self.enabled:
            return False

        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return False

        key = sys.stdin.read(1)
        return key.lower() == self.quit_key

def send_init(mega):
    if mega is None:
        return

    mega.send(Status.INIT.value)
    logger.info("%s", Status.INIT.name)

def main():

    found_count = 0
    nohead_count = 0
    last_sent_state = None
    mega = None

    try:
        with QuitKeyWatcher() as quit_key, camera.Camera() as cam:
            state = Status.NOT_FOUND
            mega = arduino.Arduino()
            #mega.send(state.value)
            last_sent_state = state
            warmup_ends_at = time.monotonic() + WARMUP_SECONDS
            stable_inference_count = 0
            slow_inference_count = 0
            controls_enabled = False
            logger.info(
                "control config found_tolerance=%s close_track=%s nohead_area=%s "
                "nohead_tolerance=%s",
                FOUND_TOLERANCE,
                CLOSE_TRACK,
                NOHEAD_AREA,
                NOHEAD_TOLERANCE,
            )
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
                if quit_key.pressed():
                    send_init(mega)
                    break

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
                    nohead_count = 0
                    state = Status.NOT_FOUND
                    logger.debug(
                        "controls blocked elapsed_warmup=%s stable_inference=%s/%s "
                        "slow_inference=%s/%s inference_ms=%.1f",
                        time.monotonic() >= warmup_ends_at,
                        stable_inference_count,
                        WARMUP_STABLE_FRAMES,
                        slow_inference_count,
                        SLOW_INFERENCE_TOLERANCE,
                        inference_ms,
                    )
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
                        nohead_count = 0
                        state = Status.NOT_FOUND
                    elif is_close_nohead_target(target):
                        nohead_count += 1
                        if nohead_count >= NOHEAD_TOLERANCE:
                            state = Status.NOHEAD
                            logger.info(
                                "%s error=%s nohead_count=%s/%s%s",
                                state.name,
                                error,
                                nohead_count,
                                NOHEAD_TOLERANCE,
                                describe_target(target),
                            )
                        else:
                            state = Status.TRACK
                            logger.info(
                                "NOHEAD pending error=%s nohead_count=%s/%s%s",
                                error,
                                nohead_count,
                                NOHEAD_TOLERANCE,
                                describe_target(target),
                            )
                    elif abs(error) <= CLOSE_TRACK:
                        nohead_count = 0
                        if is_close_enough_target(target) :
                            state = Status.CLOSE_ENOUGH
                            logger.info("%s error=%s%s", state.name, error, describe_target(target))
                        else:
                            state = Status.TRACK
                            logger.info("%s error=%s%s", state.name, error, describe_target(target))
                    else:
                        nohead_count = 0
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
                            if state == Status.CLOSE_ENOUGH:
                                time.sleep(ARM_CATCH_TIME)
                            elif state == Status.NOHEAD:
                                cam.reset_tracking()
                                found_count = 0
                                nohead_count = 0
                                time.sleep(NOHEAD_SLEEP_TIME)

                else:
                    found_count = 0
                    nohead_count = 0
                    state = Status.NOT_FOUND
                    mega.send(state.value)
                    if last_sent_state != state:
                        last_sent_state = state
                        logger.info("%s", state.name)
    finally:
        if mega is not None:
            mega.close()

if __name__ == "__main__":
    setup_logging()
    main()
