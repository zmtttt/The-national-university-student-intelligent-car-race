#!/usr/bin/python3
# -*- coding: utf-8 -*-
from joystick import JoyStick
from cart import Cart
import time
import cv2
import threading
import json
import config
#本文件用作无人驾驶车道数据采集
#本文件采用手柄巡线
cart = Cart()
class Logger:

    def __init__(self):
        self.camera = cv2.VideoCapture(config.front_cam)
        self.started = False
        self.stopped_ = False
        self.counter = 0
        self.map = {}
        self.result_dir = "/run/media/sda1"

    def start(self):
        self.started = True
        cart.steer(0)
        pass

    def stop(self):
        if self.stopped_:
            return
        self.stopped_ = True
        cart.stop()
        path = "{}/result.json".format(self.result_dir)
        with open(path, 'w') as fp:
            json.dump(self.map.copy(), fp)
            
        pass

    def log(self, axis):
        if self.started :
            t1=time.time()
            print("axis:{}".format(axis))
            #t2=time.time()
            #print('time:{.5f}',t2-t1,'counter',self.counter)
            cart.steer(axis)
            #return_value, image = self.camera.read()
            #path = "{}/{}.jpg".format(self.result_dir, self.counter)
            self.map[self.counter] = axis
            #cv2.imwrite(path, image)
            self.counter = self.counter + 1
            t2=time.time()
            print('time:{.5f}',t2-t1,'counter',self.counter)
            
    def stopped(self):
        return self.stopped_
js = JoyStick()
logger = Logger()
def joystick_thread():
    js.open()
    while not logger.stopped():
        time, value, type_, number = js.read()
        if js.type(type_) == "button":
            #print("button:{} state: {}".format(number, value))
            if number == 6 and value == 1:
                logger.start()
            if number == 7 and value == 1:
                logger.stop()
        if js.type(type_) == "axis":
            #print("axis:{} state: {}".format(number, value))
            if number == 2:
                # handle_axis(time, value)
                js.x_axis = value * 1.0 / 32767/1.5
def main():
    t = threading.Thread(target=joystick_thread, args=())
    t.start()
    cart.velocity=-20
    # logger.start()
    while not logger.stopped():
        t1=time.time()
        time.sleep(0.01)
        logger.log(js.x_axis)
    t2=time.time()
    print('time',t2-t1)
    t.join()
    cart.stop()

if __name__ == "__main__":
    main()
