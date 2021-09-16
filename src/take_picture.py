import cv2
import numpy as np
import datetime
import time
from widgets import Button
from obstacle import Lightwork
#摄像头编号
# cam=0
cam=1
#程序开启运行开关
start_button = Button(1, "UP")
#程序关闭开关
stop_button = Button(1, "DOWN")
camera = cv2.VideoCapture(cam)
camera .set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera .set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
btn=0
if __name__ == "__main__":
    if cam==0:
        result_dir="./front_image"
    #cam=1
    else:
        result_dir = "./side_image"
    
    print("Start!")
    bnt=0
    print('''Press the "Down button" to take photos!''')
    while True:
        if stop_button.clicked():
            break
        else:    
            print("btn",btn)
            path = "{}/{}.png".format('/run/media/sda1/side_4', btn);
            btn+=1
            #time.sleep(0.2)
            return_value, image = camera.read()
            name = "{}.png".format(btn)
            cv2.imwrite(path, image)
    del(camera)

