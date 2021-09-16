import cv2
import numpy as np
import predictor_wrapper
import config

cnn_args = {
    "shape": [1, 3, 128, 128],
    "ms": [125.5, 0.00392157]
}

cruise_model = config.cruise["model"]

# CNN网络的图片预处理
def process_image(frame, size, ms):
    frame = cv2.resize(frame, (size, size))
    img = frame.astype(np.float32)
    img = img - ms[0]
    img = img * ms[1]
    img = np.expand_dims(img, axis=0)
    return img;

# CNN网络预处理
def cnn_preprocess(args, img, buf):
    shape = args["shape"]
    img = process_image(img, shape[2], args["ms"]);
    hwc_shape = list(shape)
    hwc_shape[3], hwc_shape[1] = hwc_shape[1], hwc_shape[3]
    data = buf
    img = img.reshape(hwc_shape)
    # print("hwc_shape:{}".format(hwc_shape))
    data[0:, 0:hwc_shape[1], 0:hwc_shape[2], 0:hwc_shape[3]] = img
    data = data.reshape(shape)
    return data

# CNN网络预测
def infer_cnn(predictor, buf, image):
    data = cnn_preprocess(cnn_args, image, buf)
    predictor.set_input(data, 0)
    predictor.run()
    out = predictor.get_output(0)
    return np.array(out)[0][0]

class Cruiser:
    def __init__(self):
        hwc_shape = list(cnn_args["shape"])
        hwc_shape[3], hwc_shape[1] = hwc_shape[1], hwc_shape[3]
        self.buf = np.zeros(hwc_shape).astype('float32')
        self.predictor = predictor_wrapper.PaddleLitePredictor()
        self.predictor.load(cruise_model)

    def cruise(self, frame):
        res = infer_cnn(self.predictor,self.buf, frame);
        # print(res)
        return res;

if __name__ == "__main__":
    c = Cruiser();