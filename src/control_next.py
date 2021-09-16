#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import datetime
import time
import cv2
import config
from widgets import Button, Magneto_sensor, Servo,UltrasonicSensor
from widgets import *
from camera import Camera
from driver import Driver, SLOW_DOWN_RATE
from detectors import SignDetector
from detectors import TaskDetector
from detectors import in_centered_in_image
from fixed_queue import FixedQueue
from obstacle import raiseflag,shot_target,capture_target,Lightwork
#是否进行标志和目标物检测
enable_detection = True
#前置摄像头
front_camera = Camera(config.front_cam, [640, 480])
#侧边摄像头
side_camera = Camera(config.side_cam, [640, 480])
#程序开启运行开关
start_button = Button(1, "UP")
stop_button = Button(1, "DOWN")
ultr_sensor=UltrasonicSensor(4)

servo1 = Servo(1)
servo2 = Servo(3)
#车道巡航
driver = Driver()
#地面标志检测
sign_detector = SignDetector()
#侧边目标物检测
task_detector = TaskDetector()

STATE_IDLE = "idle"
STATE_CRUISE = "cruise"
STATE_SIGN_DETECTED = "sign_detected"
STATE_TASK = "task"
STATE_TASK_8="go_task8"
MISS_DURATION = 600
SLOW_DOWN_TIME = 3
TEMP_STOP_TIME = 5
handled_taskes = set()

#
OBSTACLE = 0
#存储离目标较近的目标序列
task_queue_ = FixedQueue()
#candidate队列用类存储远一点的目标
candicate_queue_ = FixedQueue()

# for mession
mession_queue = FixedQueue(class_num=5)
#任务次序记录
taskorder=0
#存储视野中较近的检测到的目标
def task_queue():
    return task_queue_;

#相较task_queue队列，candidate队列用类存储远一点的目标
def candicate_queue():
    return candicate_queue_;

#筛选距离车体最近的标志和目标物
def select_queue(detect_res, blow_index, status):
    global task_queue_, candicate_queue_

    # no object detected
    if len(detect_res) == 0:
        task_queue_.append(0)
        candicate_queue_.append(0)
    count = 0
    for item in detect_res:
        label = item.index
        # 距离较远的框不插入？
        y_bar = 0.1
        if item.index < 3:  # red,yellow,and green;
            y_bar = 0.2
        if item.index == 7:  # obstacle;
            y_bar = 0.7
        if item.relative_box[3] > y_bar:
            if count == blow_index and status == 'cruise':
                task_queue_.append(1, label)
            else:
                candicate_queue_.append(1, label)
        count += 1

#交换集合序列
def switch_queue():
    global task_queue_, candicate_queue_
    task_queue_, candicate_queue_ = candicate_queue_, task_queue_
#打印已识别任务序列
def debug_queues():
    print("task queue is ")
    print(task_queue_.deque)
    print("candidate queue is")
    print(candicate_queue_.deque)
#确认"DOWN"按键是否按下，程序是否处于等待直行状态
def check_stop(current_state):
    if current_state != STATE_IDLE and stop_button.clicked():
        return True
    return False
#任务程序入口函数
def idle_handler(arg):
    while True:
        if start_button.clicked():
            time.sleep(2)
            time.sleep(0.3)
            return STATE_CRUISE, None
        # time.sleep(0.1);
        print("IDLE")
        driver.stop()
    return STATE_IDLE, None

# 规则中应该限制相邻两个标签的距离,中心距离大于3 / 4图像长度
# 摄像头视野frame图像中最多只包含两个目标物。
def cruise_handler(arg):
    # 任务完成标志（全局变量）
    # 任务标记量主要是为了误识别导致重复做任务
    global taskorder
    # counter =0
    flagnum=0
    oldtime=time.time()
    # 设置小车巡航速度
    driver.set_speed(driver.full_speed)
    if arg != None:
        start_time = time.time()
        cur_speed = driver.full_speed
        driver.set_speed(cur_speed * SLOW_DOWN_RATE)
    if taskorder == 1:
        driver.set_Kx(0.75)
    elif taskorder==2 or taskorder==4 :
        driver.set_Kx(0.9)
    elif taskorder==7:
        driver.set_Kx(0.75)
    elif taskorder==8:
        driver.set_Kx(0.75)
        driver.set_speed(30)
    else:
        driver.set_Kx(0.85)
    while True:
        if taskorder==7:
            return STATE_TASK_8,"dingxiangjun"
        # recover speed
        if arg != None :
            cur_time = time.time()
            if cur_time - start_time > SLOW_DOWN_TIME:
                driver.set_speed(driver.full_speed)
            else:
                driver.set_speed(cur_speed * SLOW_DOWN_RATE)
        if check_stop(STATE_CRUISE):
            return STATE_IDLE, None
        # print("cruise")
        front_image = front_camera.read()
        # cv2.imwrite("front/{}.png".format(counter), front_image)
        # counter = counter + 1
        driver.go(front_image)
        if not enable_detection:
            continue
        if taskorder==8:
            continue
        # 侦测车道上有无标志图标
        res, blow_index = sign_detector.detect(front_image, "cruise")
        # sign valid maybe task maybe just signal (bluesign, triangle, light)
        # 获取标志识别结果，获得所在列表的索引值
        flag, index = task_queue().roadsign_valid()
        if flag:
            flagnum+=1
            if flagnum<2:
                continue
            else :
                if res and len(res) > 0:
                    select_queue(res, blow_index, "cruise")
                flagnum=0
            sign_name = config.sign_list[index]
            print("====")
            print(sign_name)
            print(type(sign_name))
            print(handled_taskes)
            if sign_name not in handled_taskes:
                if sign_name in ["barracks", "fenglangjuxu", "fortress", "soldier", "target"]:
                    if taskorder>=2 and taskorder<5 and sign_name!="target":
                        task_queue().clear()
                        continue
                    return STATE_SIGN_DETECTED, sign_name
                else:
                    driver.set_speed(driver.full_speed)
                    print("cruise else mode {}".format(sign_name))
                    task_queue().clear()
                    continue
        if res and len(res) > 0:
            select_queue(res, blow_index, "cruise")

#地面图标识别
def sign_detected_handler(arg):
    global taskorder
    # handled_taskes.add(arg);
    cur_speed = driver.full_speed
    driver.set_speed(cur_speed * SLOW_DOWN_RATE)
    # driver.set_speed(cur_speed * 0.5)
    miss_mission = 0
    print("sign detected")
    print(arg)
    # imgnum = 0
    barracksflag = True
    barracksnum=0
    frontimagenum=0
    disappearnum=0
    while True:
        if taskorder==7:
            return STATE_TASK_8,"dingxiangjun"
        if check_stop(STATE_SIGN_DETECTED):
            return STATE_IDLE, None
        front_image = front_camera.read()
        driver.go(front_image)
        print("sign_detected")
        res_front, blow_index = sign_detector.detect(front_image, "cruise")
        # end_add
        if res_front and len(res_front) > 0 :
            select_queue(res_front, blow_index, "cruise")
            frontimagenum+=1
            if res_front[0].name=="barracks":
                barracksnum += 1
                print("barracksnum=",barracksnum)
                if barracksnum > 25:
                    barracksflag =False
                    barracksnum=0
                    frontimagenum=0
                    return STATE_TASK, "barracks"
                else:
                    pass
        # roadsign disappear
        else:
            print("barracksnum=", barracksnum)
            if barracksnum>12:
                disappearnum+=1
                if disappearnum+barracksnum>25:
                    barracksflag = False
                    barracksnum = 0
                    frontimagenum = 0
                    return STATE_TASK, "barracks"
            else:
                pass
        if taskorder == 2:
            if frontimagenum > 5:
                driver.set_speed(cur_speed * SLOW_DOWN_RATE)
            else:
                driver.set_speed(cur_speed * 0.7)
            side_image = side_camera.read()
            # time.sleep(0.01)
            res = task_detector.detect(side_image)
        elif taskorder==4 and frontimagenum>10:
            if frontimagenum<15:
                driver.set_speed(cur_speed * 0.7)
            else:
                driver.set_speed(cur_speed * SLOW_DOWN_RATE)
            side_image = side_camera.read()
            # time.sleep(0.01)
            res = task_detector.detect(side_image)
        elif frontimagenum>15 or arg == "fenglangjuxu":
            if arg=="fenglangjuxu":
                driver.set_speed(cur_speed * 0.8)
            else:
                driver.set_speed(cur_speed * SLOW_DOWN_RATE)
            side_image = side_camera.read()
            # time.sleep(0.01)
            res = task_detector.detect(side_image)
            # print("*****************sidecam=",res)
        else:
            driver.set_speed(cur_speed*0.8)
            continue
        if taskorder<6 and res!=None and len(res)>0 and  res[0].name=="trophies":
            continue
         #视觉数据采集
        # imgnum+=1
        # print("image%d"%imgnum)
        # cv2.imwrite('./image/{}.png'.format(imgnum), side_image)
        if len(res) > 0 :
            miss_mission = 0
            if in_centered_in_image(res) :
                print("stepping into task")
                # new_add
                task_queue().clear()
                # end_add
                driver.stop()
                time.sleep(1)
                #后续右侧任务
                # return STATE_CRUISE, None
                frontimagenum=0
                print("+++++++++++++++++++start task!res=",res)
                return STATE_TASK, res
        else:
            # no object detected in this frame
            # print("detected miss")
            miss_mission += 1
        # mession searching failed return road
        if miss_mission > MISS_DURATION / driver.get_min_speed():
            print("detected miss, stepping into cruise")
            task_queue().clear()
            # end_add
            switch_queue()
            driver.set_speed(cur_speed)
            return STATE_CRUISE, None

#做任务
def task_handler(res):
    global taskorder
    print("task")
    print("res=",res)
    cur_speed = driver.full_speed
    driver.set_speed(cur_speed)
    if res=="barracks":
        driver.stop()
        time.sleep(0.5)
        driver.driver_run(-19,-20)
        time.sleep(3.4)
        driver.driver_run(-18, -8)
        time.sleep(2.0)
        driver.driver_run(-18, -8)
        time.sleep(3.0)
        driver.stop()
        for i in range(0,4):
            Lightwork(2,"red")
            time.sleep(0.2)
            Lightwork(2,"off")
        driver.driver_run(18, 8)
        time.sleep(2.8)
        driver.driver_run(18, 18)
        time.sleep(0.6)
        driver.driver_run(8, 18)
        time.sleep(2.8)
        driver.stop()
        taskorder =6
    elif res=="dingxiangjun":
        setmotor1 = Motor_rotate(4)
        driver.driver_run(18, 18)
        time.sleep(1)
        driver.stop()
        time.sleep(0.5)
        raiseflag(4, 3)
        setmotor1.motor_rotate(17)
        time.sleep(0.18)
        setmotor1.motor_rotate(0)
        time.sleep(0.1)
        taskorder =8
    else:
        name = res[0].name
        if name == "daijun":
            time.sleep(1)
            driver.driver_run(-15,-15)
            time.sleep(1)
            driver.stop()
            time.sleep(0.5)
            raiseflag(4,3)
            taskorder =1
        elif name=="dingxiangjun":
            time.sleep(1)
            driver.driver_run(-15, -15)
            time.sleep(0.5)
            driver.stop()
            time.sleep(0.5)
            raiseflag(4,3)
            taskorder =8
        elif name=="dunhuang":
            if taskorder==0:
                time.sleep(1)
                driver.driver_run(-15, -15)
                time.sleep(1)
                driver.stop()
                taskorder =1
            else:
                driver.driver_run(15, 10)
                time.sleep(1)
                driver.stop()
                taskorder = 2
            time.sleep(0.5)
            raiseflag(4,3)

        elif name=="target":
            if taskorder==2:
                time.sleep(0.7)
                driver.driver_run(-15, -8)
                time.sleep(0.3)
                driver.stop()
                time.sleep(0.3)
            elif taskorder==4:
                time.sleep(0.7)
                driver.driver_run(8, 8)
                time.sleep(0.2)
                driver.stop()
                time.sleep(0.3)
            shot_target(2)
            taskorder += 1
        elif name=="trophies":
            driver.driver_run(-10,-10)
            time.sleep(1.2)
            driver.stop()
            time.sleep(1)
            capture_target(3, 2)
            taskorder =7
            time.sleep(0.7)
        else:
            print("Error!#####################")

    #右侧任务
    if taskorder == 1 or taskorder == 2 or taskorder == 3 or taskorder == 4 :
        servo1.servocontrol(-123, 40)
    # 后续左侧任务
    else:
        servo1.servocontrol(43, 40)
    time.sleep(2)
    switch_queue()
    return STATE_CRUISE, None
def go_task_8(arg):
    global taskorder
    print("go_task_8")
    cur_speed = driver.full_speed
    driver.set_speed(cur_speed)
    task8_start=time.time()
    while True:
        front_image = front_camera.read()
        driver.go(front_image)
        if time.time()-task8_start<4:
            driver.set_Kx(0.7)
            continue
        elif time.time()-task8_start<5:
            driver.set_Kx(0.8)
        else:
            driver.set_Kx(0.9)
        um = ultr_sensor.read()
        # print("*8888888")
        if um != None and um < 20:
            driver.stop()
            return STATE_TASK, "dingxiangjun"

state_map = {
    STATE_IDLE: idle_handler,
    STATE_CRUISE: cruise_handler,
    STATE_SIGN_DETECTED: sign_detected_handler,
    STATE_TASK: task_handler,
    STATE_TASK_8:go_task_8,
}

def main():

    motor = Motor_rotate(2)
    time.sleep(0.2)
    #左侧
    servo1.servocontrol(40, 30)
    #右侧
    # servo1.servocontrol(-127, 40)
    time.sleep(1)
    servo2.servocontrol(-85, 100)
    time.sleep(2)
    current_state = STATE_IDLE
    # current_state = STATE_CRUISE
    arg = None
    front_camera.start()
    side_camera.start()
    time.sleep(0.2)
    Lightwork(2, "red")
    time.sleep(0.5)
    Lightwork(2, "green")
    time.sleep(0.5)
    Lightwork(2, "off")
    try:
        while (True):
            new_state, arg = state_map[current_state](arg)
            current_state = new_state
        driver.stop()
        front_camera.stop()
        side_camera.stop()
    except ZeroDivisionError as e:
        print('except:', e)
    finally:
        print('finally...')
        Lightwork(2, "off")
        Lightwork(4, "off")

if __name__ == "__main__":
    main()
