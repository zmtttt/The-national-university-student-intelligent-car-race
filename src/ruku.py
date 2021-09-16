from driver import Driver
import json
from camera import Camera
from detectors import SignDetector
import cv2
import time
from cart import Cart
driver = Driver()
cart = Cart()
front_camera = Camera(0, [640, 480])
sign_detector = SignDetector()
if __name__ == '__main__':
    #for i in range(20):
        #front_image=front_camera.read()
        #front_image[:160,:]=0
        #res,index = sign_detector.detect(front_image)
        #print('res:',res)
        #print('index',index)
    time.sleep(0.5)
    driver.driver_run(-10,-15)
    time.sleep(7.6)
    #driver.driver_run(-18, -8)
    #time.sleep(3.0)
    driver.stop()
    time.sleep(1.0)
    driver.driver_run(10, 15)
    time.sleep(5.8)
    driver.stop()
