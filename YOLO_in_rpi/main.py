import os
import logging
import time
from enum import Enum
import arduino
import camera_YOLO as camera

class Status(Enum):
    TRACK = 0
    NOT_FOUND = 1
    CLOSE_ENOUGH = 2
    OUT_OF_BOUND = 3
    IDLE = 4

FOUND_TOLERANCE = 2         
CLOSE_TRACK = 20
CLOSE_AREA = 30000           

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

def main():

    found_count = 0
    last_sent_state = None

    try:
        with camera.Camera() as cam:
            state = Status.NOT_FOUND
            mega = arduino.Arduino()
            mega.send(state.value)
            last_sent_state = state

            for ball_detected, error, target in cam.streaming():
                mega.print_all(mega.receive())
                if ball_detected:
                    found_count += 1

                    if found_count < FOUND_TOLERANCE:
                        continue

                    if error is None:
                        state = Status.IDLE
                    elif abs(error) <= CLOSE_TRACK:
                        if has_target(target) and target["area"] >= CLOSE_AREA:
                            state = Status.CLOSE_ENOUGH
                        else:
                            state = Status.TRACK
                    else:
                        state = Status.TRACK

                    if state == Status.TRACK:
                        mega.send(f"{state.value} {error}")
                        last_sent_state = state
                        logger.info("%s error=%s", state.name, error)
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
