  #!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import datetime
import time
#import copy as cp
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
    
#import threading
#threading.Thread(target=serial.write(bytes.fromhex('77 68 06')), args=()).start()
#77 68 06
#threading.Thread(target=serial.write(bytes.fromhex('77 68 06')), args=()).start()
#import threading
balance=True
def serial_keep():
    global balance
    while balance:
        serial.write(bytes.fromhex('77 68 06'))
        #time.sleep(0.5)

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
    global balance
    import threading
    threading.Thread(target=serial.write(bytes.fromhex('77 68 06')), args=()).start()
    #my_thread=threading.Thread(target=serial_keep())
    #my_thread.start()
    #立起来
    servo2_zhi.servocontrol(-85,50)
    time.sleep(0.5)
    #闭合
    servo2.servocontrol(30, 50)
    time.sleep(1)
    servo2_zhi.servocontrol(-25,20)
    time.sleep(1)
        #抓取
    servo2.servocontrol(120, 50)
    time.sleep(2)
        #举起来
    servo2_zhi.servocontrol(-85,50)
    time.sleep(2)
    balance=False
    #stop_thread(my_thread)
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
    time.sleep(0.8)
    motor2.motor_rotate(0)

def daoqiu():
    #以下为倒球
    motor1.motor_rotate(10)
    time.sleep(1)
    motor1.motor_rotate(-25)
    time.sleep(0.1)
    #落下
    motor1.motor_rotate(10)
    time.sleep(1)
    #举起
    motor1.motor_rotate(-25)
    time.sleep(0.1)
    motor1.motor_rotate(0)

def shengqi():
    temp=0
    
    #motor4.motor_rotate(-20)
    motor4.motor_rotate(-30)
    #for i in range(10):
    res=magsens.read()
    #time.sleep(0.2)  
    #time.sleep(1.5)
    #motor4.motor_rotate(0)
    #t1=time.time()  
    while True:
        #time.sleep(2)
        #motor4.motor_rotate(0)
        res=magsens.read()
        print("res=",res)
        if res>=99:
            motor4.motor_rotate(0)
            #time.sleep(1)
            break
    #t2=time.time()
    #print('time:',t2-t1)
    for i in range(0,3):
        #t1=time.time()
        Lightwork(2, "green")
        #t2=time.time()
        time.sleep(0.05)
        Lightwork(2, "off")
        time.sleep(0.05)
    print('----------')
    #motor4.motor_rotate(-20)
    #for i in range(20):
        #res=magsens.read()
    #motor4.motor_rotate(-13)
    #time.sleep(0.1)
    #motor4.motor_rotate(0)
    #time.sleep(0.1)

def check_stop():
    if stop_button.clicked():
        return True
    return False


mask_dict={3:shengqi,5:jiba,2:zhuaqu}


if __name__ == '__main__':
    
    sign_detector=SignDetector()
    while True:
      shengqi()
   