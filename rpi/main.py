import os
import time
import logging
from enum import Enum
import psutil
import arduino
# import camera
#test camera
import cameraUI as camera

MISMATCH_TOLERANCE = 5
class Status(Enum):
    ERROR = 0
    NOT_FOUND = 1
    CLOSE_ENOUGH = 2
    OUT_OF_BOUND = 3
    IDLE = 4

MISMATCH_TOLERANCE = 5      
FOUND_TOLERANCE = 2         
CLOSE_ERROR = 20           

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


def main():
    mega = arduino.Arduino()

    found_count = 0
    last_sent_state = None

    try:
        with camera.Camera() as cam:
            state = Status.NOT_FOUND
            mega.send(state.value)
            last_sent_state = state

            for ball_detected, error in cam.streaming():
                if ball_detected:
                    found_count += 1

                    if found_count < FOUND_TOLERANCE:
                        continue

                    if error is None:
                        state = Status.IDLE
                    elif abs(error) <= CLOSE_ERROR:
                        state = Status.CLOSE_ENOUGH
                    else:
                        state = Status.ERROR

                    if state == Status.ERROR:
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
