import serial
import serial.tools.list_ports
import time


class Arduino:
    def __init__(self, baudrate=9600, timeout=1):
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.setup()

    def find_arduino(self):
        ports = serial.tools.list_ports.comports()
        keywords = ["arduino", "ch340", "usb serial"]

        for port in ports:
            desc = port.description.lower()

            if any(k in desc for k in keywords):
                return port.device

        return None

    def setup(self):
        arduino_port = self.find_arduino()

        if not arduino_port:
            print("arduino not found")
            return

        print("found:", arduino_port)

        self.ser = serial.Serial(
            arduino_port,
            self.baudrate,
            timeout=self.timeout
        )

        time.sleep(2)

    def send(self, msg):
        if self.ser is None:
            #print("serial not connected")
            return

        if not msg.endswith("\n"):
            msg += "\n"

        self.ser.write(msg.encode())

    def receive(self, wait_time=0.5):
        if self.ser is None:
            print("serial not connected")
            return None

        start = time.time()

        while time.time() - start < wait_time:
            if self.ser.in_waiting > 0:
                msg = self.ser.readline().decode(errors="ignore").strip()
                return msg

            time.sleep(0.01)

        return None

    def close(self):
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()