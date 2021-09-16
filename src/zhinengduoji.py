import threading
import time

from serial_port import serial_connection
serial = serial_connection

class Servo:
    def __init__(self, ID):
        self.ID = ID
        self.ID_str = '{:02x}'.format(ID)

    def servocontrol(self, angle, speed):
        while True:
            if self.stopped:
                return
            print('1')
            cmd_servo_data = bytes.fromhex('77 68 06 00 02 36') + bytes.fromhex(self.ID_str) + speed.to_bytes(1,
                                                                                                            byteorder='big', \
                                                                                                            signed=True) + angle.to_bytes(
            1, byteorder='big', signed=True) + bytes.fromhex('0A')

        # for i in range(0,2):
        serial.write(cmd_servo_data)
        #     time.sleep(0.3)
        
    def start(self):
        threading.Thread(target=self.servocontrol, args=()).start()
        
    def stop(self):
        self.stopped = True
print()