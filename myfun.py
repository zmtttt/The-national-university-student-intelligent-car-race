#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import copy
import time
import math
threshold=120
def del_color(image):
    color = [
        ([15, 50, 50], [27, 255, 255])  # 黄色范围~这个是我自己试验的范围，可根据实际情况自行调整~注意：数值按[b,g,r]排布
    ]
    #color=[([20,43,46],[45,255,255])]
    # 如果color中定义了几种颜色区间，都可以分割出来
    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限

        # 根据阈值找到对应颜�?

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # 展示图片
        # cv2.imshow("images", np.hstack([image, output]))
        # cv2.waitKey(0)
    return output



def calculation(x):
    if(x>0):
        #number=1/(1+2.7^((-x+200)/50))*4.8-0.07
        number=2.74*(x/180)*(x/180)
        return number
    elif x<0:
        number=-2.74*(x/180)*(x/180)
        #number=-(1/(1+2.7^((x+200)/50))*4.8-0.07)
        return number
    else: return 0

def pd(x,last_x):
    #global last_x
    p = 1.05
    d = 0.098
    #p=1.1
    #d=0.1
    differ=(x*p+d*(x-last_x))/120
    last_x=x
    return differ,last_x

def find_center(image,last_x):
    sum_value=np.zeros((640,))
    left,right=0,640
    #cv2.line(image,(0,320-1),(640,320-1),(255,255))
    #cv2.line(image, (0, 360+1), (640, 360+1), (255, 255))
    for i in image[340:356]:
        sum_value+=np.array(i)
    sum_value=sum_value/15
    #print(sum_value)
    a=[True,True]
    s=300
    for i in range(298):
        if sum_value[s-i-1]  > threshold and a[0]:
            left=s-i-1
            a[0]=False
        if sum_value[s+i+1] > threshold and a[1]:
            right=s+i+1
            a[1]=False
        if(not a[0]) and (not a[1]):
            break
    center=(left+right)/2
    #print(left,right,a)
    #if(left==0 and right !=640):
    #    differ=revise_road(image,right)
    #    return differ
    #elif(left!=0 and right ==640):
    #    differ=revise_road(image,left)
    #    return differ
    #cv2.line(image,(0,260),(640,260),(255,255))
    #cv2.line(image, (0, 285), (640, 285), (255, 255))
    #cv2.line(image, (320,0 ), (320, 640), (255, 255))
    #if(abs(center-350)<20):
    #    return 0,0
    #if(right==640 and left==0):
        #differ=cross(image)
    #return 0,0
    #differ = (center-310)/320*1.4
    #differ = calculation(center-350)
    differ,last_x= pd(center-s,last_x)
    return differ,last_x

def _main(image1,last_x):
    #global last_x
    image=copy.deepcopy(image1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image=del_color(image)
    #image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    differ,last_x1=find_center(image[:,:,2],last_x)
    return differ,image,last_x1

if __name__ == '__main__':
    cap=cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #print(fps,h,w)
    vid_writer = cv2.VideoWriter('D:/Temp/point/123.avi', fourcc, fps, (w, h))
    while True:
        flag,image=cap.read()
        #print('2')
        if flag==False:
            print('not find')
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image=del_color(image)
        image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        differ=find_center(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        vid_writer.write(image)
        cv2.imshow('123',image)
    cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()
