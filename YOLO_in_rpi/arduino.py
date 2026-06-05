import serial
import serial.tools.list_ports
import time
import logging
import os

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

class Arduino:
    def __init__(self, baudrate=9600, timeout=0):
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.setup()

    def find_arduino(self):
        ports = serial.tools.list_ports.comports()
        keywords = ["arduino", "ch340", "usb serial"]

        for port in ports:
            desc = port.description.lower()
            logger.debug("port device=%s desc=%s hwid=%s", port.device, port.description, port.hwid)

            if any(k in desc for k in keywords):
                return port.device

        return None

    def setup(self):
        arduino_port = self.find_arduino()

        if not arduino_port:
            logger.warning("arduino not found")
            return

        logger.info("found arduino: %s", arduino_port)

        self.ser = serial.Serial(arduino_port, self.baudrate, timeout=self.timeout)

        time.sleep(2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        logger.info("serial connected baudrate=%s", self.baudrate)
        self.send("rdy")
        time.sleep(0.3)

    def send(self, msg):
        if self.ser is None:
            logger.debug("serial not connected, skip send: %s", msg)
            return
        msg = str(msg)
        if not msg.endswith("\n"):
            msg += "\n"

        self.ser.write(msg.encode())
        logger.info("[SEND][Arduino]: %s", msg.strip())

    def receive(self, wait_time=0.5):
        if self.ser is None:
            logger.warning("serial not connected")
            return []

        messages = []
        while self.ser.in_waiting > 0:
            msg = self.ser.readline().decode(errors="ignore").strip()
            if msg:
                messages.append(msg)
                logger.info("[GET][Arduino]: %s", msg)
        return messages
        
    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None
            logger.info("serial closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    setup_logging()
    mega = Arduino()
    if mega.ser is None:
        raise SystemExit(1)

    logger.info("sending test message")
    mega.send("Hi")
    logger.info("waiting for Arduino messages... Press Ctrl+C to quit.")
    try:
        while True:
            messages = mega.receive()
            for msg in messages:
                print(msg)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        mega.close()
