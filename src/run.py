#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import datetime
import time
import cv2
import json
import config
from widgets import Button, LimitSwitch, Infrared_value, Servo
from widgets import *
from camera import Camera
from driver import Driver, SLOW_DOWN_RATE
from detectors import SignDetector
from detectors import TaskDetector
from detectors import in_centered_in_image
from fixed_queue import FixedQueue
from obstacle import *
from cruiser import Cruiser
from cart import Cart
front_camera = Camera(0, [640, 480])
side_camera = Camera(1, [640, 480])
driver = Driver()
cruiser = Cruiser()
#程序开启运行开关
start_button = Button(1, "UP")
#程序关闭开关
stop_button = Button(1, "DOWN")

#确认"DOWN"按键是否按下，程序是否处于等待直行状态
def check_stop():
    if stop_button.clicked():
        return True
    return False
if __name__=='__main__':
    front_camera.start()
    side_camera.start()
    #基准速度
    driver.set_speed(-30)
    #转弯系数
    driver.cart.Kx=0.9
    #延时
    time.sleep(0.5)
    while True:
        if start_button.clicked():
            time.sleep(0.3)
            break
        print("Wait for start!")
    counter=0
    result_dir ='/run/media/sda1/xunxian'
    map1={}
    while True:
        front_image = front_camera.read()
        #front_image = front_image[::-1,::-1]
        #side_image = side_camera.read()
        #side_image = side_image[::-1,::-1]
        angle=driver.go(front_image)
        path = "{}/{}.jpg".format(result_dir, counter)
        cv2.imwrite(path, front_image)
        #print(angle)
        map1[counter] = float(angle)
        if check_stop():
            driver.stop()
            print("End of program!")
            break
        counter+=1
    path = "{}/result.json".format(result_dir)
    with open(path, 'w') as fp:
        json.dump(map1.copy(), fp)
    front_camera.stop()
    side_camera.stop()