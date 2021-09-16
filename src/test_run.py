#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import datetime
import time
import cv2
import config
from widgets import Button
from obstacle import Lightwork
from camera import Camera
from driver import Driver, SLOW_DOWN_RATE
from detectors import SignDetector
from detectors import TaskDetector
from detectors import in_centered_in_image
#是否进行标志和目标物检测
enable_detection = False
#前置摄像头
front_camera = Camera(config.front_cam, [640, 480])
#侧边摄像头
side_camera = Camera(config.side_cam, [640, 480])
driver = Driver()
#程序开启运行开关
start_button = Button(1, "UP")
#程序关闭开关
stop_button = Button(1, "DOWN")

STATE_IDLE = "idle"
STATE_CRUISE = "cruise"
#确认"DOWN"按键是否按下，程序是否处于等待直行状态
def check_stop(current_state):
    if current_state != STATE_IDLE and stop_button.clicked():
        return True
    return False

#任务程序入口函数
def idle_handler(arg):
    while True:
        if start_button.clicked():
            time.sleep(0.3)
            return STATE_CRUISE, None
        print("IDLE")
        driver.stop()
    return STATE_IDLE, None

def cruise_handler(arg):
    #设置小车巡航速度
    driver.set_speed(driver.full_speed)
    if arg != None:
        start_time = time.time()
        cur_speed = driver.full_speed
        driver.set_speed(cur_speed * SLOW_DOWN_RATE * 0.8)
    while True:
        if arg != None:
            cur_time = time.time()
            if cur_time - start_time > SLOW_DOWN_TIME:
                driver.set_speed(driver.full_speed)
        if check_stop(STATE_CRUISE):
            return STATE_IDLE, None
        print("cruise")
        front_image = front_camera.read()
        driver.go(front_image)

state_map = {
    STATE_IDLE: idle_handler,
    STATE_CRUISE: cruise_handler
}

def main():
    front_camera.start()
    side_camera.start()
    time.sleep(0.2)
    Lightwork(2, "red")
    Lightwork(4, "red")
    time.sleep(0.2)
    Lightwork(2, "green")
    Lightwork(4, "green")
    time.sleep(0.2)
    Lightwork(2, "off")
    Lightwork(4, "off")
    current_state = STATE_IDLE
    arg = None
    while (True):
        print(current_state)
        new_state, arg = state_map[current_state](arg)
        current_state = new_state

    driver.stop()
    front_camera.stop()
    side_camera.stop()

if __name__=='__main__':
    main()