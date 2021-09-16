# -*- coding: utf-8 -*-

import sys
import json
import datetime

import cv2
import numpy as np
import datetime

import predictor_wrapper

engine = "PaddleLite"
THRESHOLD = 0.5

cnn_args = {
    "shape": [1, 3, 128, 128]
}

yolo_args = {
    "shape": [1, 3, 480, 480]
}

def draw_boxes(img, valid_results):
    """ draw SSD boxes on image"""
    res = list(img.shape)
    # for item in valid_results:
    #     if item[2] > 1 or item[3] > 1 or item[4] > 1 or item[5] > 1:
    #         # YOLO network don't need to multiply original dimenstion;
    #         res[0] = 1
    #         res[1] = 1
    #         break
    for _, item in enumerate(valid_results):
        print(item,THRESHOLD)
        if item[1] < THRESHOLD:
            continue;
        print(item[1])
        left = item[2] * res[1]
        top = item[3] * res[0]
        right = item[4] * res[1]
        bottom = item[5] * res[0]
        start_point = (int(left), int(top))
        end_point = (int(right), int(bottom))
        color = (204, 0, 204)
        thickness = 2
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    cv2.imwrite("result.jpg", img)


def dataset(frame, size):
    frame = cv2.resize(frame, (size, size))
    lower_hsv = np.array([25, 75, 190])
    upper_hsv = np.array([40, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    img = mask
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    print("image_shape:", img.shape)
    return img;

def preprocess(args, img):
    """ preprocess image using formula y = (x - mean) x scale """
    shape = args["shape"]
    print(shape)
    #img = dataset(img, shape[2]);

    shape = args["shape"]
    hwc_shape = list(shape)
    hwc_shape[3], hwc_shape[1] = hwc_shape[1], hwc_shape[3]
    data = np.zeros(hwc_shape).astype('float32')
    img = img.reshape(hwc_shape)
    data[0:, 0:hwc_shape[1], 0:hwc_shape[2], 0:hwc_shape[3]] = img
    if engine == "PaddlePaddle":
        data = data.transpose(0, 3, 1, 2)  # PaddlePaddle CHW;
    data = data.reshape(shape)
    return data

def yolo_preprocess(args, src):
    shape = args["shape"]
    img = cv2.resize(src, (shape[3], shape[2]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img -= 127.5
    img *= 0.007843

    z = np.zeros((1, shape[2], shape[3], 3)).astype(np.float32)
    z[0, 0:img.shape[0], 0:img.shape[1] + 0, 0:img.shape[2]] = img
    z = z.reshape(1, 3, shape[3], shape[2]);
    #print(z)
    return z;

def infer_cnn(predictor, image):
    data = preprocess(cnn_args, image)
    predictor.set_input(data, 0)
    predictor.run()
    out = predictor.get_output(0)
    print(out.data())

def infer_tiny_yolo(predictor, image):
    a = datetime.datetime.now()
    times = 100
    data = yolo_preprocess(yolo_args, image)
    print(data.shape)
    ####cv2.imwrite("./1.jpg",data)

    b = datetime.datetime.now()
    c = b - a
    print("proprocess used:{} ms".format(c.microseconds / 1000 / times))
    print(data.shape)
    predictor.set_input(data, 0)

    print(image.shape)

    dims = np.array([image.shape[0], image.shape[1]]).astype("int32")
    dims = dims.reshape([1, 2])
    print('dims:',dims)
    predictor.set_input(dims, 1)
    
    a = datetime.datetime.now()

    times = 100
    predictor.run()

    b = datetime.datetime.now()
    c = b - a
    mill = c.microseconds / 1000.0;
    print("detection used:{} ms".format(mill / times))

    out = predictor.get_output(0)
    print(np.array(out))
    cv2.imwrite("2.jpg",image)
    print("11",image.shape)
    #draw_boxes(image, np.array(out))

def create_predictor():
    if engine == "PaddlePaddle":
        return predictor_wrapper.PaddlePaddlePredictor()
    else:
        return predictor_wrapper.PaddleLitePredictor()

def main():
    #cnn_predictor = create_predictor()
    #print(cnn_predictor)
    # cnn_predictor.load("models/w1")
    yolo_predictor = create_predictor()
    print(yolo_predictor)
    # yolo_predictor.load("models/task")
    yolo_predictor.load("models/sign")
    # sign_fuse


    image_path = "/run/media/sda1/xunxian7_7_15/2079.jpg"

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #img = cv2.imread(image_path)
    #infer_cnn(cnn_predictor, img)
    infer_tiny_yolo(yolo_predictor, img)

main()
