# 配置相关配置
import os
import paddle as paddle

import paddle.fluid as fluid
import numpy as np


path = "/home/aistudio/work/data/"
test_list = "d1.txt"
train_list = "d1.txt"
save_path = "/home/aistudio/model_infer2"

crop_size = 128
resize_size = 128


test_list = path + test_list
train_list = path + train_list

# 定义模型
def cnn_model(image):
    temp = fluid.layers.conv2d(input=image, num_filters=32, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=32, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=3, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=128, filter_size=3, stride=1, act='relu')
    # temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=3, stride=1, act='relu')
    # temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=3, stride=1, act='relu')
    temp = fluid.layers.dropout(temp, dropout_prob=0.1)
    fc1 = fluid.layers.fc(input=temp, size=128, act="leaky_relu")
    fc2 = fluid.layers.fc(input=fc1, size=32, act="leaky_relu")
    drop_fc2 = fluid.layers.dropout(fc2, dropout_prob=0.1)
    predict = fluid.layers.fc(input=drop_fc2, size=1, act=None)
    predict = fluid.layers.tanh(predict / 4)
    return predict


# 定义数据增强手段
import cv2
import random
from PIL import Image, ImageEnhance
import numpy as np


def color_filter_autumn(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_AUTUMN)
    return im_color


def color_filter_bone(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_BONE)
    return im_color


def color_filter_winter(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_WINTER)
    return im_color


def apply_hue(img):
    low, high, prob = [-18, 18, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img

    img = img.astype(np.float32)
    delta = np.random.uniform(low, high)
    u = np.cos(delta * np.pi)
    w = np.sin(delta * np.pi)
    bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                     [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                      [1.0, -1.107, 1.705]])
    t = np.dot(np.dot(ityiq, bt), tyiq).T
    img = np.dot(img, t)
    return img


def apply_saturation(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    img = img.astype(np.float32)
    gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
    gray = gray.sum(axis=2, keepdims=True)
    gray *= (1.0 - delta)
    img *= delta
    img += gray
    return img


def apply_contrast(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    img = img.astype(np.float32)
    img *= delta
    return img


def apply_brightness(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    img = img.astype(np.float32)
    img += delta
    return img


color_maps = [
              apply_hue,
              apply_saturation,
              apply_contrast,
              apply_brightness
              ]
# 定义reader
import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image
import cv2 as cv



def gen_random_ind():
    seed = random.random()
    if seed < 1 / 4:
        return 0
    elif seed >= 1 / 4 and seed < 2 / 4:
        return 1
    elif seed >= 2 / 4 and seed < 3 / 4:
        return 2
    else:
        return 3


# 训练图片的预处理
def train_mapper(sample):
    img_path, label, crop_size, resize_size = sample
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 统一图片大小
        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
        # 把图片转换成numpy值
        img = np.array(img).astype(np.float32)

        # 随机图像增强
        id = gen_random_ind()
        # id = 3
        img = color_maps[id](img)

        # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        # 转换成CHW
        img = img.transpose((2, 0, 1))
        # 转换成BGR
        img = (img[(2, 1, 0), :, :] - 125.5) / 255.0
        return img, float(label)
    except Exception as e:
        print("{} 该图片错误表， Exception:{}".format(img_path, e))


# 获取训练的reader
def train_reader(train_list_path, crop_size, resize_size):
    father_path = os.path.dirname(train_list_path)

    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱图像列表
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                img, label = line.split('\t')
                # print(line)
                img = os.path.join(father_path, img)
                if os.path.isfile(img):
                    yield img, label, crop_size, resize_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 500)


# 测试图片的预处理
def test_mapper(sample):
    img, label, crop_size = sample
    img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 统一图像大小
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = (img[(2, 1, 0), :, :] - 125.5) / 255.0
    return img, float(label)


# 测试的图片reader
def test_reader(test_list_path, crop_size):
    father_path = os.path.dirname(test_list_path)

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                if os.path.isfile(img):
                    yield img, label, crop_size

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)


# 定义数据增强手段
import cv2
import random
from PIL import Image, ImageEnhance
import numpy as np


def color_filter_autumn(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_AUTUMN)
    return im_color


def color_filter_bone(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_BONE)
    return im_color


def color_filter_winter(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_WINTER)
    return im_color


def apply_hue(img):
    low, high, prob = [-18, 18, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img

    img = img.astype(np.float32)
    delta = np.random.uniform(low, high)
    u = np.cos(delta * np.pi)
    w = np.sin(delta * np.pi)
    bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                     [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                      [1.0, -1.107, 1.705]])
    t = np.dot(np.dot(ityiq, bt), tyiq).T
    img = np.dot(img, t)
    return img


def apply_saturation(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    img = img.astype(np.float32)
    gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
    gray = gray.sum(axis=2, keepdims=True)
    gray *= (1.0 - delta)
    img *= delta
    img += gray
    return img


def apply_contrast(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    img = img.astype(np.float32)
    img *= delta
    return img


def apply_brightness(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)
    img = img.astype(np.float32)
    img += delta
    return img


color_maps = [
              apply_hue,
              apply_saturation,
              apply_contrast,
              apply_brightness
              ]
# 定义reader
import os
import random
from multiprocessing import cpu_count
import numpy as np
import paddle
from PIL import Image
import cv2 as cv



def gen_random_ind():
    seed = random.random()
    if seed < 1 / 4:
        return 0
    elif seed >= 1 / 4 and seed < 2 / 4:
        return 1
    elif seed >= 2 / 4 and seed < 3 / 4:
        return 2
    else:
        return 3


# 训练图片的预处理
def train_mapper(sample):
    img_path, label, crop_size, resize_size = sample
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # 统一图片大小
        img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
        # 把图片转换成numpy值
        img = np.array(img).astype(np.float32)

        # 随机图像增强
        #id = gen_random_ind()
        id = 3
        img = color_maps[id](img)

        # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        # 转换成CHW
        img = img.transpose((2, 0, 1))
        # 转换成BGR
        img = (img[(2, 1, 0), :, :] - 125.5) / 255.0
        return img, float(label)
    except Exception as e:
        print("{} 该图片错误表， Exception:{}".format(img_path, e))


# 获取训练的reader
def train_reader(train_list_path, crop_size, resize_size):
    father_path = os.path.dirname(train_list_path)

    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            # 打乱图像列表
            np.random.shuffle(lines)
            # 开始获取每张图像和标签
            for line in lines:
                img, label = line.split('\t')
                # print(line)
                img = os.path.join(father_path, img)
                if os.path.isfile(img):
                    yield img, label, crop_size, resize_size

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 500)


# 测试图片的预处理
def test_mapper(sample):
    img, label, crop_size = sample
    img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 统一图像大小
    img = img.resize((crop_size, crop_size), Image.ANTIALIAS)
    # 转换成numpy值
    img = np.array(img).astype(np.float32)
    # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = (img[(2, 1, 0), :, :] - 125.5) / 255.0
    return img, float(label)


# 测试的图片reader
def test_reader(test_list_path, crop_size):
    father_path = os.path.dirname(test_list_path)

    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                img = os.path.join(father_path, img)
                if os.path.isfile(img):
                    yield img, label, crop_size

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)



# 定义logger
import os
import logging


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


logger = init_log_config()
# 创建模型，优化策略，数据读取器
import os
import paddle as paddle

import paddle.fluid as fluid
import numpy as np


image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='float32')

model = cnn_model(image)

cost = fluid.layers.elementwise_sub(model, label)
cost = fluid.layers.abs(cost)
avg_cost = fluid.layers.mean(cost)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
lr = fluid.layers.piecewise_decay(boundaries=[700, 1100], values=[0.0001, 0.00001, 0.000001])
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, regularization=fluid.regularizer.L2Decay(0.00005))

opts = optimizer.minimize(avg_cost)

# 获取自定义数据
train_reader = paddle.batch(reader=train_reader(train_list, crop_size, resize_size), batch_size=2048)
test_reader = paddle.batch(reader=test_reader(test_list, crop_size), batch_size=2048)

# 定义执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 开始训练和保存模型
all_test_cost = []
for pass_id in range(210):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])

        # 每100个batch打印一次信息
        if batch_id % 1 == 0:
            logger.info('Pass:%d, Batch:%d, TrainCost:%0.5f' %
                        (pass_id, batch_id, train_cost[0]))

    # 进行测试
    test_costs = []

    for batch_id, data in enumerate(test_reader()):
        test_cost, predict = exe.run(program=test_program,
                                     feed=feeder.feed(data),
                                     fetch_list=[avg_cost, model])
        logger.info('batch test cost {}'.format(test_cost[0]))
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    # 每轮测试的最终结果保存
    all_test_cost.append(test_cost)

    logger.info('Test:%d, Cost:%0.5f' % (pass_id, test_cost))

    if min(all_test_cost) >= test_cost:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fluid.io.save_inference_model(save_path, feeded_var_names=[image.name],
                                      main_program=fluid.default_main_program(), target_vars=[model], executor=exe,
                                      params_filename='params', model_filename='models')
        logger.info('finally test_cost: {}'.format(test_cost))
