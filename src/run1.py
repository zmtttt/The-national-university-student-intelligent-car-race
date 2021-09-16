#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import datetime
import time
import cv2
import config
import json
from myfun import _main
from cart import Cart
from widgets import Button, LimitSwitch, Infrared_value, Servo
from widgets import *
from camera import Camera
#from driver import Driver, SLOW_DOWN_RATE
from detectors import SignDetector
from detectors import TaskDetector
from detectors import in_centered_in_image
from fixed_queue import FixedQueue
from obstacle import *
from cruiser import Cruiser
from cart import Cart
front_camera = Camera(0, [640, 480])
side_camera = Camera(1, [640, 480])
#driver = Driver()
cruiser = Cruiser()
#程序开启运行开�?
start_button = Button(1, "UP")
#程序关闭开�?
stop_button = Button(1, "DOWN")
cart=Cart()
#确认"DOWN"按键是否按下，程序是否处于等待直行状�?
def check_stop():
    if stop_button.clicked():
        return True
    return False
last_x=0
if __name__=='__main__':
    #front_camera.start()
    side_camera.start()
    #基准速度
    #driver.set_speed(-30)
    #转弯系数
    #driver.cart.Kx=0.9
    cart.velocity=0
    #延时
    #time.sleep(0.5)
    #while True:
        #if start_button.clicked():
            #time.sleep(0.3)
            #break
        #print("Wait for start!")
    counter=0
    map1={}
    result_dir ='/run/media/sda1/side_7'
    while True:
        #front_image = front_camera.read()
        #front_image = front_image[::-1,::-1]
        side_image = side_camera.read()
        side_image = side_image[::-1,::-1]
        #differ,image,last_x1=_main(front_image,last_x)
        #if(differ>1):
            #differ=1
        #elif(differ<-1):
            #differ=-1
        #cart.steer(differ)
        #driver.go(front_image)
        #map1[counter] = differ
        path = "{}/{}.jpg".format(result_dir, counter)
        cv2.imwrite(path, side_image)
        if check_stop():
            cart.stop()
            print("End of program!")
            break
        counter+=1
    #path = "{}/result.json".format(result_dir)
    #with open(path, 'w') as fp:
    #    json.dump(map1.copy(), fp)
    #front_camera.stop()
    side_camera.stop()
    