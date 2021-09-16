#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import datetime
import time
#import copy as cp
import threading
import cv2
import config
from collections import Counter
from widgets import Servo, Servo_pwm,Motor_rotate, Magneto_sensor,UltrasonicSensor,Light,Buzzer
from widgets import *
from camera import Camera
from driver import Driver
from detectors import SignDetector

stop_button = Button(1, "DOWN")
motor1=Motor_rotate(1)
motor2=Motor_rotate(2)
motor4=Motor_rotate(4)

servo2=Servo_pwm(2)
servo2_zhi = Servo(2)
servo1_zhi = Servo(1)
    
magsens=Magneto_sensor(3)
driver = Driver()
    


def Lightwork(light_port,color):
    light=Light(light_port)
    red=[80,0,0]
    green=[0,80,0]
    yellow=[80,80,0]
    off=[0,0,0]
    light_color=[0,0,0]
    if color =='red':
        light_color=red
    elif color=='green':
        light_color=green
    elif color=='yellow':
        light_color=yellow
    elif color=='off':
        light_color = off
    light.lightcontrol(0,light_color[0],light_color[1],light_color[2])

def zhuaqu():
    #my_thread=threading.Thread(target=serial_keep())
    #my_thread.start()
    #立起来
    servo2_zhi.servocontrol(-80,50)
    time.sleep(1)
    #闭合
    servo2.servocontrol(30, 50)
    time.sleep(1)
    #落下
    servo2_zhi.servocontrol(-20,20)
    time.sleep(1)
        #抓取
    servo2.servocontrol(120, 50)
    time.sleep(2)
        #举起来
    servo2_zhi.servocontrol(-80,60)
    time.sleep(2)
def zhuaqu1():
    t1=threading.Thread(target=serial_keep)
    t2=threading.Thread(target=zhuaqu1)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
def jiba():
    motor2.motor_rotate(20)
    time.sleep(0.8)
    #往里拉
    motor2.motor_rotate(-20)
    time.sleep(0.3)
    motor2.motor_rotate(0)

def daoqiu():
    #以下为倒球
    motor1.motor_rotate(20)
    time.sleep(0.5)
    motor1.motor_rotate(-20)
    time.sleep(0.5)
    #落下
    motor1.motor_rotate(20)
    time.sleep(0.5)
    #举起
    motor1.motor_rotate(-30)
    time.sleep(0.5)
    motor1.motor_rotate(0)

def shengqi():
    print("shengqi_start!")
    motor4.motor_rotate(-20)
    time.sleep(0.5)
    for i in range(0,10):
        res=magsens.read()
    while True:
        for i in range(0,10):
            res=magsens.read()
        if res:
            res=res
        else:
            res = 0
        print("res=",res)
        if (res and res>98):
            motor4.motor_rotate(0)
            time.sleep(1)
            break
    for i in range(0,3):
        Lightwork(2, "green")
        time.sleep(0.05)
        Lightwork(2, "off")
        time.sleep(0.05)
    print("shengqi_end!")
    
#def ruku()
def end():
    pass

def check_stop():
    if stop_button.clicked():
        return True
    return False


mask_dict={1:shengqi,2:jiba,4:zhuaqu,5:daoqiu,7:end}
#mask_dict={}


if __name__ == '__main__':
    
    sign_detector=SignDetector()
    front_camera = Camera(config.front_cam, [640, 480])
    front_camera.start()
    driver.set_speed(0)
    #servo2_zhi.servocontrol(-80,50)
    time.sleep(2)
    #import threading
    #print('1')
    #threading.Thread(target=serial.write(bytes.fromhex('77 68 06')), args=()).start()
    #print('1')
    while True:
        if check_stop():
            driver.stop()
            print("End of program!")
            break
            #time.sleep(0.5)
    for i in range(40):
        front_image = front_camera.read()
        driver.go(front_image)
    #servo2_zhi.servocontrol(-80,50)
    #time.sleep(0.5)
    driver.set_speed(-40)
    counter_detect=[]
    temp=0
    #time.sleep(1)
    while True:
        front_image = front_camera.read()
        #front_image_copy=cp.deepcopy(front_image)
        #front_image_copy[240:,:]=0
        #cv2.imwrite('image.jpg',front_image_copy)
        #if temp<10:
        #    temp+=1
        #    continue
        #front_image = front_camera.read()
        driver.go(front_image)
        front_image[:200,:]=0
        #print(front_image)
        res,index = sign_detector.detect(front_image)
        #print('res',res)
        #print('index',index)
        if(res ): 
            #print(res[0][0]) 
            if abs(res[0][2][0]+res[0][2][2]-640)<170:
                counter_detect.append(int(res[0][0]))
                print('counter_detect',counter_detect)
                print('res',res[0][0])
        elif(counter_detect):
            counter_detect.append(-1)
        if(len(counter_detect)>3):
            state=1
            for i in counter_detect[-3:]:
                if(i!=-1):
                    state=0
                    break
            if(state==1):
                if(len(counter_detect)>5):
                    #time.sleep(0.4)
                    driver.stop()
                    time.sleep(0.5)
                    re=Counter(counter_detect).most_common(1)
                    if re[0][0] in list(mask_dict.keys()):
                        if re[0][0]==6:
                            break
                        mask_dict[re[0][0]]()
                    driver.set_speed(-40)
                counter_detect=[]
        #cv2.imwrite('image',front_image)
        if check_stop():
            driver.stop()
            print("End of program!")
            break
    front_camera.stop()