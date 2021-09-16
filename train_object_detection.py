from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import uuid
import numpy as np
import time
import six
import math
import paddle
import paddle.fluid as fluid
import logging
import xml.etree.ElementTree
import codecs

from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from PIL import Image, ImageEnhance, ImageDraw

logger = None
train_parameters = {
    "input_size": [3, 300, 300],
    # 包含背景类
    "class_dim": 6,
    "label_dict": {},
    "image_count": -1,
    "log_feed_image": False,
    "pretrained": True,
    "pretrained_model_dir": "/home/aistudio/pretrained-model",
    "continue_train": False,
    "save_model_dir": "/home/aistudio/ssd-model",
    "model_prefix": "mobilenet-ssd",
    # "data_dir": "/home/work/xiangyubo/common_resource/pascalvoc/pascalvoc",
    "data_dir": "/home/aistudio/data/",
    "mean_rgb": [127.5, 127.5, 127.5],
    "file_list": "road_train.txt",
    "eval_list": "road_eval.txt",
    "mode": "train",
    "multi_data_reader_count": 5,
    "num_epochs": 120,
    "train_batch_size": 2,
    "use_gpu": False,
    "apply_distort": True,
    "apply_expand": True,
    "apply_corp": True,
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "rsm_strategy": {
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.05, 0.01],
    },
    "momentum_strategy": {
        "learning_rate": 0.1,
        "decay_steps": 2 ** 7,
        "decay_rate": 0.8
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "min_loss": 1.28,
        "min_curr_map": 0.86
    }
}
# 定义模型
class MobileNetSSD:
    def __init__(self):
        pass

    def conv_bn(self,
                input,
                filter_size,
                num_filters,
                stride,
                padding,
                num_groups=1,
                act='relu',
                use_cudnn=True):
        parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, num_filters1, num_filters2, num_groups, stride, scale):
        depthwise_conv = self.conv_bn(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False)

        pointwise_conv = self.conv_bn(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)
        return pointwise_conv

    def extra_block(self, input, num_filters1, num_filters2, num_groups, stride, scale):
        # 1x1 conv
        pointwise_conv = self.conv_bn(
            input=input,
            filter_size=1,
            num_filters=int(num_filters1 * scale),
            stride=1,
            num_groups=int(num_groups * scale),
            padding=0)

        # 3x3 conv
        normal_conv = self.conv_bn(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2 * scale),
            stride=2,
            num_groups=int(num_groups * scale),
            padding=1)
        return normal_conv

    def net(self, num_classes, img, img_shape, scale=1.0):
        # 300x300
        tmp = self.conv_bn(img, 3, int(32 * scale), 2, 1)
        # 150x150
        tmp = self.depthwise_separable(tmp, 32, 64, 32, 1, scale)
        tmp = self.depthwise_separable(tmp, 64, 128, 64, 2, scale)
        # 75x75
        tmp = self.depthwise_separable(tmp, 128, 128, 128, 1, scale)
        tmp = self.depthwise_separable(tmp, 128, 256, 128, 2, scale)
        # 38x38
        tmp = self.depthwise_separable(tmp, 256, 256, 256, 1, scale)
        tmp = self.depthwise_separable(tmp, 256, 512, 256, 2, scale)

        # 19x19
        for i in range(5):
            tmp = self.depthwise_separable(tmp, 512, 512, 512, 1, scale)
        module11 = tmp
        tmp = self.depthwise_separable(tmp, 512, 1024, 512, 2, scale)

        # 10x10
        module13 = self.depthwise_separable(tmp, 1024, 1024, 1024, 1, scale)
        module14 = self.extra_block(module13, 256, 512, 1, 2, scale)
        # 5x5
        module15 = self.extra_block(module14, 128, 256, 1, 2, scale)
        # 3x3
        module16 = self.extra_block(module15, 128, 256, 1, 2, scale)
        # 2x2
        module17 = self.extra_block(module16, 64, 128, 1, 2, scale)

        mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
            inputs=[module11, module13, module14, module15, module16, module17],
            image=img,
            num_classes=num_classes,
            min_ratio=20,
            max_ratio=90,
            min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
            max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.], [2., 3.]],
            base_size=img_shape[2],
            offset=0.5,
            flip=True)

        return mbox_locs, mbox_confs, box, box_var
# 定义训练时候，数据增强需要的辅助类，例如外接矩形框、采样器
class sampler:
    def __init__(self, max_sample, max_trial, min_scale, max_scale,
                 min_aspect_ratio, max_aspect_ratio, min_jaccard_overlap,
                 max_jaccard_overlap):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap


class bbox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

def init_train_parameters():
    file_list = os.path.join(train_parameters['data_dir'], train_parameters["file_list"])
    label_list = os.path.join(train_parameters['data_dir'], "label_list")
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)


def init_log_config():
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
# 训练数据增强，主要是采样。利用随机截取训练图上的框来生成新的训练样本。
#同时要保证采样的样本能包含真实的目标。采样之后，为了保持训练数据格式的一致性，还需要对标注的坐标信息做变换
def log_feed_image(img, sampled_labels):
    draw = ImageDraw.Draw(img)
    target_h = train_parameters['input_size'][1]
    target_w = train_parameters['input_size'][2]
    for label in sampled_labels:
        print(label)
        draw.rectangle((label[1] * target_w, label[2] * target_h, label[3] * target_w, label[4] * target_h), None,
                       'red')
    img.save(str(uuid.uuid1()) + '.jpg')


def bbox_area(src_bbox):
    width = src_bbox.xmax - src_bbox.xmin
    height = src_bbox.ymax - src_bbox.ymin
    return width * height


def generate_sample(sampler):
    scale = np.random.uniform(sampler.min_scale, sampler.max_scale)
    aspect_ratio = np.random.uniform(sampler.min_aspect_ratio, sampler.max_aspect_ratio)
    aspect_ratio = max(aspect_ratio, (scale ** 2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale ** 2.0))

    bbox_width = scale * (aspect_ratio ** 0.5)
    bbox_height = scale / (aspect_ratio ** 0.5)
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = bbox(xmin, ymin, xmax, ymax)
    return sampled_bbox


def jaccard_overlap(sample_bbox, object_bbox):
    """
    计算交并比
    """
    if sample_bbox.xmin >= object_bbox.xmax or \
            sample_bbox.xmax <= object_bbox.xmin or \
            sample_bbox.ymin >= object_bbox.ymax or \
            sample_bbox.ymax <= object_bbox.ymin:
        return 0
    intersect_xmin = max(sample_bbox.xmin, object_bbox.xmin)
    intersect_ymin = max(sample_bbox.ymin, object_bbox.ymin)
    intersect_xmax = min(sample_bbox.xmax, object_bbox.xmax)
    intersect_ymax = min(sample_bbox.ymax, object_bbox.ymax)
    intersect_size = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
    if sampler.min_jaccard_overlap == 0 and sampler.max_jaccard_overlap == 0:
        return True
    for i in range(len(bbox_labels)):
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2], bbox_labels[i][3], bbox_labels[i][4])
        overlap = jaccard_overlap(sample_bbox, object_bbox)
        if sampler.min_jaccard_overlap != 0 and overlap < sampler.min_jaccard_overlap:
            continue
        if sampler.max_jaccard_overlap != 0 and overlap > sampler.max_jaccard_overlap:
            continue
        return True
    return False


def generate_batch_samples(batch_sampler, bbox_labels):
    sampled_bbox = []
    index = []
    c = 0
    for sampler in batch_sampler:
        found = 0
        for i in range(sampler.max_trial):
            if found >= sampler.max_sample:
                break
            sample_bbox = generate_sample(sampler)
            if satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
                sampled_bbox.append(sample_bbox)
                found = found + 1
                index.append(c)
        c = c + 1
    return sampled_bbox


def clip_bbox(src_bbox):
    src_bbox.xmin = max(min(src_bbox.xmin, 1.0), 0.0)
    src_bbox.ymin = max(min(src_bbox.ymin, 1.0), 0.0)
    src_bbox.xmax = max(min(src_bbox.xmax, 1.0), 0.0)
    src_bbox.ymax = max(min(src_bbox.ymax, 1.0), 0.0)
    return src_bbox

def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox.xmax + src_bbox.xmin) / 2
    center_y = (src_bbox.ymax + src_bbox.ymin) / 2
    if center_x >= sample_bbox.xmin and \
            center_x <= sample_bbox.xmax and \
            center_y >= sample_bbox.ymin and \
            center_y <= sample_bbox.ymax:
        return True
    return False


def transform_labels(bbox_labels, sample_bbox):
    """
    裁剪之后，坐标要发生相应变化
    """
    proj_bbox = bbox(0, 0, 0, 0)
    sample_labels = []
    for i in range(len(bbox_labels)):
        sample_label = []
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2], bbox_labels[i][3], bbox_labels[i][4])
        if not meet_emit_constraint(object_bbox, sample_bbox):
            continue
        sample_width = sample_bbox.xmax - sample_bbox.xmin
        sample_height = sample_bbox.ymax - sample_bbox.ymin
        proj_bbox.xmin = (object_bbox.xmin - sample_bbox.xmin) / sample_width
        proj_bbox.ymin = (object_bbox.ymin - sample_bbox.ymin) / sample_height
        proj_bbox.xmax = (object_bbox.xmax - sample_bbox.xmin) / sample_width
        proj_bbox.ymax = (object_bbox.ymax - sample_bbox.ymin) / sample_height
        proj_bbox = clip_bbox(proj_bbox)
        if bbox_area(proj_bbox) > 0:
            sample_label.append(bbox_labels[i][0])
            sample_label.append(float(proj_bbox.xmin))
            sample_label.append(float(proj_bbox.ymin))
            sample_label.append(float(proj_bbox.xmax))
            sample_label.append(float(proj_bbox.ymax))
            sample_label.append(bbox_labels[i][5])
            sample_labels.append(sample_label)
    return sample_labels


def crop_image(img, bbox_labels, sample_bbox, image_width, image_height):
    """
    裁剪图片
    """
    sample_bbox = clip_bbox(sample_bbox)
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)
    sample_img = img.crop((xmin, ymin, xmax, ymax))
    sample_labels = transform_labels(bbox_labels, sample_bbox)
    return sample_img, sample_labels
# 图像增强
def resize_img(img, sampled_labels):
    """
    缩放图片
    """
    target_size = train_parameters['input_size']
    ret = img.resize((target_size[1], target_size[2]), Image.ANTIALIAS)
    return ret


def random_brightness(img):
    """
    随机调整亮度
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_distort_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    """
    随机调整对比度
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_distort_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    """
    随机调整饱和度
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_distort_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    """
    随机颜色
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['hue_prob']:
        hue_delta = train_parameters['image_distort_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    else:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def expand_image(img, bbox_labels, img_width, img_height):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['expand_prob']:
        expand_max_ratio = train_parameters['image_distort_strategy']['expand_max_ratio']
        if expand_max_ratio - 1 >= 0.01:
            expand_ratio = np.random.uniform(1, expand_max_ratio)
            height = int(img_height * expand_ratio)
            width = int(img_width * expand_ratio)
            h_off = math.floor(np.random.uniform(0, height - img_height))
            w_off = math.floor(np.random.uniform(0, width - img_width))
            expand_bbox = bbox(-w_off / img_width, -h_off / img_height,
                               (width - w_off) / img_width,
                               (height - h_off) / img_height)
            expand_img = np.uint8(np.ones((height, width, 3)) * np.array([127.5, 127.5, 127.5]))
            expand_img = Image.fromarray(expand_img)
            expand_img.paste(img, (int(w_off), int(h_off)))
            bbox_labels = transform_labels(bbox_labels, expand_bbox)
            return expand_img, bbox_labels, width, height
    return img, bbox_labels, img_width, img_height


def preprocess(img, bbox_labels, mode):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    if mode == 'train':
        if train_parameters['apply_distort']:
            img = distort_image(img)
        if train_parameters['apply_expand']:
            img, bbox_labels, img_width, img_height = expand_image(img, bbox_labels, img_width, img_height)

        if train_parameters['apply_corp']:
            batch_sampler = []
            # hard-code here
            batch_sampler.append(sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0))
            sampled_bbox = generate_batch_samples(batch_sampler, bbox_labels)
            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                img, sampled_labels = crop_image(img, bbox_labels, sampled_bbox[idx], img_width, img_height)

        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for i in six.moves.xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp

    img = resize_img(img, sampled_labels)
    if train_parameters['log_feed_image']:
        log_feed_image(img, sampled_labels)
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sampled_labels
# 数据读取器，模型构建，优化策略
def trainval_reader(file_list, data_dir, mode):
    def reader():
        file_list = os.path.join(data_dir, train_parameters['file_list'])
        with open(file_list) as f:
            file_list = f.readlines()

        np.random.shuffle(file_list)
        for line in file_list:
            line = line.strip()
            image_path, label_path = line.split()
            image_path = os.path.join(data_dir, image_path)
            label_path = os.path.join(data_dir, label_path)
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            im_width, im_height = img.size
            # layout: label | xmin | ymin | xmax | ymax | difficult
            bbox_labels = []
            root = xml.etree.ElementTree.parse(label_path).getroot()
            for object in root.findall('object'):
                bbox_sample = []
                # start from 1
                bbox_sample.append(float(train_parameters['label_dict'][object.find('name').text]))
                bbox = object.find('bndbox')
                difficult = float(object.find('difficult').text)
                bbox_sample.append(float(bbox.find('xmin').text) / im_width)
                bbox_sample.append(float(bbox.find('ymin').text) / im_height)
                bbox_sample.append(float(bbox.find('xmax').text) / im_width)
                bbox_sample.append(float(bbox.find('ymax').text) / im_height)
                bbox_sample.append(difficult)
                bbox_labels.append(bbox_sample)
            img, sample_labels = preprocess(img, bbox_labels, mode)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            difficults = sample_labels[:, -1].astype('int32')
            yield img, boxes, lbls, difficults

    return paddle.batch(reader=reader, batch_size=train_parameters["train_batch_size"])


def test_reader(file_list, data_dir, mode):
    def reader():
        np.random.shuffle(file_list)
        for line in file_list:
            if mode == 'train' or mode == 'eval':
                image_path, label_path = line.split()
                image_path = os.path.join(data_dir, image_path)
                label_path = os.path.join(data_dir, label_path)
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                # layout: label | xmin | ymin | xmax | ymax | difficult
                bbox_labels = []
                root = xml.etree.ElementTree.parse(label_path).getroot()
                for object in root.findall('object'):
                    bbox_sample = []
                    # start from 1
                    bbox_sample.append(float(train_parameters['label_dict'][object.find('name').text]))
                    bbox = object.find('bndbox')
                    difficult = float(object.find('difficult').text)
                    bbox_sample.append(float(bbox.find('xmin').text) / im_width)
                    bbox_sample.append(float(bbox.find('ymin').text) / im_height)
                    bbox_sample.append(float(bbox.find('xmax').text) / im_width)
                    bbox_sample.append(float(bbox.find('ymax').text) / im_height)
                    bbox_sample.append(difficult)
                    bbox_labels.append(bbox_sample)
                img, sample_labels = preprocess(img, bbox_labels, mode)
                sample_labels = np.array(sample_labels)
                if len(sample_labels) == 0: continue
                boxes = sample_labels[:, 1:5]
                lbls = sample_labels[:, 0].astype('int32')
                difficults = sample_labels[:, -1].astype('int32')
                yield img, boxes, lbls, difficults
            elif mode == 'test':
                img_path = os.path.join(data_dir, line)
                yield Image.open(img_path)

    return reader


def multi_process_custom_reader(file_path, data_dir, num_workers, mode):
    file_path = os.path.join(data_dir, file_path)
    readers = []
    images = [line.strip() for line in open(file_path)]
    n = int(math.ceil(len(images) // num_workers))
    image_lists = [images[i: i + n] for i in range(0, len(images), n)]
    for l in image_lists:
        readers.append(paddle.batch(custom_reader(l, data_dir, mode),
                                    batch_size=train_parameters['train_batch_size'],
                                    drop_last=True))
    return paddle.reader.multiprocess_reader(readers, False)


def create_eval_reader(file_path, data_dir, mode):
    file_path = os.path.join(data_dir, file_path)
    images = [line.strip() for line in open(file_path)]
    return paddle.batch(custom_reader(images, data_dir, mode),
                        batch_size=train_parameters['train_batch_size'],
                        drop_last=True)


def build_train_program_with_async_reader(main_prog, startup_prog, place):
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[4], dtype='float32', lod_level=1)
        gt_label = fluid.layers.data(name='gt_label', shape=[1], dtype='int32', lod_level=1)
        difficult = fluid.layers.data(name='difficult', shape=[1], dtype='int32', lod_level=1)
        # data_reader = fluid.layers.create_py_reader_by_data(capacity=64,
        #                                                     feed_list=[img, gt_box, gt_label, difficult],
        #                                                     name='train')
        data_reader = trainval_reader(train_parameters['file_list'], train_parameters['data_dir'], mode='train')
        # multi_reader = multi_process_custom_reader(train_parameters['file_list'],
        #                                            train_parameters['data_dir'],
        #                                            train_parameters['multi_data_reader_count'],
        #                                            'train')
        # data_reader.decorate_paddle_reader(multi_reader)
        with fluid.unique_name.guard():
            # img, gt_box, gt_label, difficult = fluid.layers.read_file(data_reader)
            feeder = fluid.DataFeeder(place=place, feed_list=[img, gt_box, gt_label, difficult])
            model = MobileNetSSD()
            locs, confs, box, box_var = model.net(train_parameters['class_dim'], img, train_parameters['input_size'])
            with fluid.unique_name.guard('train'):
                loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box, box_var)
                loss = fluid.layers.reduce_sum(loss)
                optimizer = optimizer_rms_setting()
                optimizer.minimize(loss)
                return data_reader, img, loss, locs, confs, box, box_var, feeder


def build_eval_program_with_feeder(main_prog, startup_prog):
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[4], dtype='float32', lod_level=1)
        gt_label = fluid.layers.data(name='gt_label', shape=[1], dtype='int32', lod_level=1)
        difficult = fluid.layers.data(name='difficult', shape=[1], dtype='int32', lod_level=1)
        feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label, difficult], place=place, program=main_prog)
        # reader = create_eval_reader(train_parameters['file_list'], train_parameters['data_dir'], 'eval')
        data_reader = trainval_reader(train_parameters['eval_list'], train_parameters['data_dir'], mode='eval')
        with fluid.unique_name.guard():
            model = MobileNetSSD()
            locs, confs, box, box_var = model.net(train_parameters['class_dim'], img, train_parameters['input_size'])
            with fluid.unique_name.guard('eval'):
                nmsed_out = fluid.layers.detection_output(locs, confs, box, box_var, nms_threshold=0.45)
                map_eval = fluid.metrics.DetectionMAP(nmsed_out, gt_label, gt_box, difficult,
                                                      train_parameters['class_dim'], overlap_threshold=0.5,
                                                      evaluate_difficult=False, ap_version='11point')
                cur_map, accum_map = map_eval.get_map_var()
                return feeder, data_reader, cur_map, accum_map, nmsed_out


def optimizer_momentum_setting():
    learning_strategy = train_parameters['momentum_strategy']
    learning_rate = fluid.layers.exponential_decay(learning_rate=learning_strategy['learning_rate'],
                                                   decay_steps=learning_strategy['decay_steps'],
                                                   decay_rate=learning_strategy['decay_rate'])
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.1)
    return optimizer


def optimizer_rms_setting():
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    learning_strategy = train_parameters['rsm_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005))

    return optimizer


def save_model(base_dir, base_name, feed_var_list, target_var_list, train_program, infer_program, exe):
    fluid.io.save_persistables(dirname=base_dir,
                               filename=base_name + '-retrain',
                               main_program=train_program,
                               executor=exe)
    fluid.io.save_inference_model(dirname=base_dir,
                                  params_filename=base_name + '-params',
                                  model_filename=base_name + '-model',
                                  feeded_var_names=feed_var_list,
                                  target_vars=target_var_list,
                                  main_program=infer_program,
                                  executor=exe)


def load_pretrained_params(exe, program):
    retrain_param_file = os.path.join(train_parameters['save_model_dir'],
                                      train_parameters['model_prefix'] + '-retrain')
    if os.path.exists(retrain_param_file) and train_parameters['continue_train']:
        logger.info('load param from retrain model')
        print('load param from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_model_dir'],
                                   main_program=program,
                                   filename=train_parameters['model_prefix'] + '-retrain')
    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_model_dir']):
        logger.info('load param from pretrained model')
        print('load param from pretrained model')

        def if_exist(var):
            return os.path.exists(os.path.join(train_parameters['pretrained_model_dir'], var.name))

        fluid.io.load_vars(exe, train_parameters['pretrained_model_dir'], main_program=program,
                           predicate=if_exist)
init_log_config()
init_train_parameters()
print("start ssd, train params:", str(train_parameters))
logger.info("start ssd, train params: %s", str(train_parameters))

logger.info("create place, use gpu:" + str(train_parameters['use_gpu']))
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()

logger.info("build network and program")
train_program = fluid.Program()
start_program = fluid.Program()
eval_program = fluid.Program()
start_program = fluid.Program()
train_reader, img, loss, locs, confs, box, box_var, train_feeder = build_train_program_with_async_reader(train_program, start_program, place)
eval_feeder, eval_reader, cur_map, accum_map, nmsed_out = build_eval_program_with_feeder(eval_program, start_program)
# eval_program = eval_program.clone(for_test=True)  # 注意设置 test，不然 batch_normal 之类的参数会不固化

logger.info("build executor and init params")
exe = fluid.Executor(place)
exe.run(start_program)
train_fetch_list = [loss.name]
eval_fetch_list = [cur_map.name, accum_map.name]
load_pretrained_params(exe, train_program)

stop_strategy = train_parameters['early_stop']
successive_limit = stop_strategy['successive_limit']
sample_freq = stop_strategy['sample_frequency']
min_accu_map = 0
min_loss = stop_strategy['min_loss']
stop_train = False
total_batch_count = 0
successive_count = 0
for pass_id in range(train_parameters["num_epochs"]):
    logger.info("current pass: %d, start read image", pass_id)
    batch_id = 0
    # train_reader.start()
    for batch_id, data in enumerate(train_reader()):
        t1 = time.time()
        train_cost = exe.run(program=train_program,
                            feed=train_feeder.feed(data),
                            fetch_list=train_fetch_list)
        period = time.time() - t1
        loss = np.mean(np.array(train_cost))
        batch_id += 1
        if batch_id % 10 == 0:
            logger.info(
                "Pass {0}, trainbatch {1}, loss {2} time {3}".format(pass_id, batch_id, loss, "%2.2f sec" % period))
            print(
                "Pass {0}, trainbatch {1}, loss {2} time {3}".format(pass_id, batch_id, loss, "%2.2f sec" % period))
        # # just for test
        # break
        # # end test
    # train_reader.reset()
    for data in eval_reader():
        cur_map_v, accum_map_v = exe.run(eval_program, feed=eval_feeder.feed(data),
                                         fetch_list=eval_fetch_list)
        # break
        logger.info(
            "{0} batch train, cur_map:{1} accum_map_v:{2}".format(total_batch_count, cur_map_v[0],
                                                                           accum_map_v[0]))
        print("{0} batch train, cur_map:{1} accum_map_v:{2}".format(total_batch_count, cur_map_v[0],
                                                                             accum_map_v[0]))
    if accum_map_v[0] > min_accu_map:
        min_accu_map = accum_map_v[0]
        fluid.io.save_inference_model(dirname=train_parameters['save_model_dir'],
                                      params_filename=train_parameters['model_prefix'] + '-params',
                                      model_filename=train_parameters['model_prefix'] + '-model',
                                      feeded_var_names=['img'],
                                      target_vars=[nmsed_out],
                                      main_program=eval_program,
                                      executor=exe)
logger.info('End Training')
