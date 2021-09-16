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
    front_camera.start()
    temp = 0
    while True:
        temp = temp+1
        front_image=front_camera.read()
        front_image[:160,:]=0
        res,index = sign_detector.detect(front_image)
        time.sleep(0.1)
        print('jjj',front_image.shape)
        print('res:',res)
        print('index',index)
        print('temp',temp)