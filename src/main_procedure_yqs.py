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
    #threading.Thread(target=serial.write(bytes.fromhex('77 68 06')), args=()).start()
    #time.sleep(1)
    #print('zhuaqu')
    #servo1_zhi.servocontrol(113,50)
    #time.sleep(1.5)
    #立起来
    servo2_zhi.servocontrol(-80,50)
    time.sleep(1)
    #闭合
    servo2.servocontrol(30, 50)
    time.sleep(1)
    #落下
    servo2_zhi.servocontrol(-15,20)
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
    servo1_zhi.servocontrol(-45,50)
    time.sleep(1.5)
    motor2.motor_rotate(20)
    time.sleep(0.8)
    #往里拉
    motor2.motor_rotate(-20)
    time.sleep(0.15)
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
        if (res and res>98):
            motor4.motor_rotate(0)
            time.sleep(1)
            break
    for i in range(0,3):
        t1=time.time()
        Lightwork(2, "green")
        t2=time.time()
        time.sleep(0.05)
        if(t2-t1<0.5):
            time.sleep(0.6)
        Lightwork(2, "off")
        time.sleep(0.05)
        if(t2-t1<0.5):
            time.sleep(0.6)

def ruku():
    global if_dosign1
    driver.driver_run(16,16)
    time.sleep(1.9)
    driver.driver_run(-5,-15)
    time.sleep(3.8)
    driver.driver_run(-16, -15)
    time.sleep(1.8)
    driver.stop()
    time.sleep(1.0)
    driver.driver_run(14, 15)
    time.sleep(3.0)
    driver.driver_run(3, 15)
    time.sleep(2.4)
    driver.stop()
    if_dosign1=1

def end():
    pass

def check_stop():
    if stop_button.clicked():
        return True
    return False



mask_dict={1:shengqi,2:jiba,4:zhuaqu,5:daoqiu,3:ruku,6:end}


if __name__ == '__main__':
    if_dosign1=0
    sign_detector=SignDetector()
    front_camera = Camera(config.front_cam, [640, 480])
    front_camera.start()
    driver.set_speed(0)
    time.sleep(2)
    while True:
        if check_stop():
            driver.stop()
            print("End of program!")
            break
            #time.sleep(0.5)
    for i in range(30):
        front_image = front_camera.read()
        driver.go(front_image)
    #servo2_zhi.servocontrol(-80,50)
    #time.sleep(0.5)
    driver.set_speed(-40)
    counter_detect=[]
    temp=0
    while True:
        front_image = front_camera.read()
        #front_image = front_camera.read()
        driver.go(front_image)
        front_image[:220,:]=0
        
        res,index = sign_detector.detect(front_image)
        #print(res[0][0])
        if(res ): 
            #print(res[0][0]) 
            if abs(res[0][2][0]+res[0][2][2]-640)<185:
                counter_detect.append(int(res[0][0]))
                if(int(res[0][0])==4 and if_dosign1==1):
                    if_dosign1=0
                    counter_detect=[]
                    counter_detect.append(int(res[0][0]))
                    print(res[0][0])
                
        elif(counter_detect):
            counter_detect.append(-1)
        if(len(counter_detect)>3 and if_dosign1==0):
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
                    print(re[0][0],re)
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