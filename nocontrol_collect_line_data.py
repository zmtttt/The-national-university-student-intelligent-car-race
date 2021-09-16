#!/usr/bin/python3
# -*- coding: utf-8 -*-
from joystick import JoyStick
from cart import Cart
import time
import cv2
import threading
import json
import config
from myfun import _main
#本文件用作无人驾驶车道数据采集
#本文将使用不使用手柄采集
cart = Cart()
class Logger:

    def __init__(self):
        self.camera = cv2.VideoCapture(config.front_cam)
        self.started = False
        self.stopped_ = False
        self.counter = 3000
        self.map = {}
        self.last_x=0
        self.result_dir = "/run/media/sda1/xunxian7_8_14/"

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
            #print("axis:".format(axis))
            #cart.steer(axis)
            return_value, image = self.camera.read()
            differ,_,self.last_x=_main(image,self.last_x)
            if differ>1:
              differ=1
            elif differ<-1:
              differ=-1
            cart.steer(differ)
            print('differ',differ)
            path = "{}/{}.jpg".format(self.result_dir, self.counter)
            self.map[self.counter] = differ
            cv2.imwrite(path, image)
            self.counter = self.counter + 1
            
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
            print("axis:{} state: {}".format(number, value))
            if number == 2:
                # handle_axis(time, value)
                js.x_axis = value * 1.0 / 32767
def main():
    t = threading.Thread(target=joystick_thread, args=())
    t.start()
    cart.velocity=-25
    # logger.start()
    while not logger.stopped():
        # time.sleep(0.01)
        logger.log(js.x_axis)

    t.join()
    cart.stop()

if __name__ == "__main__":
    main()
