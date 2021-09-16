import os
import cv2
from cruiser import Cruiser
from classifier import Classifier
from detectors import SignDetector,TaskDetector
from camera import Camera
import config
import time
#测试图片存放位置和测试输出结果位置
cruiser_images_dir = "test/cruise"
cruiser_result_dir = "test/cruise_res"
task_images_dir = "test/task"
task_result_dir = "test/task_res"
sign_images_dir = "test/sign"
sign_result_dir = "test/sign_res"
front_image_dir = "test/front"
front_result_dir = "test/front_res"

image_extensions = [".png",".jpg",".jpeg"]

def read_dir(dir_path):
    files = []
    for filename in os.listdir(dir_path):
        file_name, file_extension = os.path.splitext(filename)
        if (file_extension.lower() in image_extensions):
            files.append(filename)
    return files

def save_image(frame, save_path):
    cv2.imwrite(save_path, frame)

def draw_cruise_result(frame, res):
    color = (0, 244, 10)
    thickness = 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = 450, 50

    fontScale = 1
    txt = "{:.4f}".format(round(res, 5))
    frame = cv2.putText(frame, txt, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
    print("angle=",txt)
    return frame

def draw_res(frame, results):
    res = list(frame.shape)
    print(results)
    for item in results:
        print(item)
        print(type(item))
        left = item.relative_box[0] * res[1]
        top = item.relative_box[1] * res[0]
        right = item.relative_box[2] * res[1]
        bottom = item.relative_box[3] * res[0]
        start_point = (int(left), int(top))
        end_point = (int(right), int(bottom))
        color = (0, 244, 10)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = start_point[0], start_point[1] - 10
        fontScale = 1
        frame = cv2.putText(frame, item.name, org, font,
                           fontScale, color, thickness, cv2.LINE_AA)
        return frame
#对前向摄像头拍摄的视频文件进行模型推理。
def test_front_video():
    #视频文件
    cap=cv2.VideoCapture('run.avi')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)
    cruiser = Cruiser()
    sd = SignDetector()
    time.sleep(1)
    time_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./' + time_name + '.avi', fourcc, 6, (640, 480))
    while True:
        ret,front_image = cap.read()
        cruise_result = cruiser.cruise(front_image )
        frame = draw_cruise_result(front_image , cruise_result)
        signs, index = sd.detect(frame)
        draw_res(frame, signs)
        # frame = cv2.flip(frame, 2)
        out.write(frame)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
#对前向摄像头拍摄图片进行动态识别，包括车道值，不同的车道值代表了不同的转弯强度
def test_cruise():
    front_camera = Camera(config.front_cam, [640, 480])
    front_camera.start()
    time.sleep(1)
    cruiser = Cruiser()
    time.sleep(1)
    while True:
        front_image = front_camera.read()
        cruise_result = cruiser.cruise(front_image )
        frame = draw_cruise_result(front_image , cruise_result)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    front_camera.stop()
    cv2.destroyAllWindows()
#前向车道地面标志动态识别
def test_sign():
    front_camera = Camera(config.front_cam, [640, 480])
    front_camera.start()
    time.sleep(1)
    cruiser = Cruiser()
    sd = SignDetector()
    time.sleep(1)
    while True:
        front_image = front_camera.read()
        cruise_result = cruiser.cruise(front_image )
        frame = draw_cruise_result(front_image , cruise_result)
        signs, index = sd.detect(frame)
        draw_res(frame, signs)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    front_camera.stop()
    cv2.destroyAllWindows()
#侧向任务动态识别
def test_task():
    side_camera = Camera(config.side_cam, [640, 480])
    side_camera.start()
    time.sleep(1)
    td = TaskDetector()
    time.sleep(1)
    while True:
        side_image = side_camera.read()
        tasks = td.detect(side_image)
        draw_res(side_image, tasks)
        cv2.imshow("Output", side_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    side_camera.stop()
    cv2.destroyAllWindows()

#前向车道和地面标志图片识别
def test_front():
    cruiser = Cruiser()
    sd = SignDetector()
    files = read_dir(front_image_dir)
    for filename in files:
        file_path = os.path.join(front_image_dir, filename)
        frame = cv2.imread(file_path)
        cruise_result = cruiser.cruise(frame)
        frame = draw_cruise_result(frame, cruise_result)
        signs,index = sd.detect(frame)
        save_path = os.path.join(front_result_dir, filename)
        draw_res(frame, signs)
        save_image(frame, save_path)

#侧向任务模型图片识别
def test_task_detector():
    td = TaskDetector()
    files = read_dir(task_images_dir)
    for filename in files:
        file_path = os.path.join(task_images_dir, filename)
        frame = cv2.imread(file_path)
        tasks = td.detect(frame)
        save_path = os.path.join(task_result_dir, filename)
        draw_res(frame, tasks)
        save_path(frame, save_path)

#前向车道地面标志识别
def test_sign_detector():
    sd = SignDetector()
    files = read_dir(sign_images_dir)
    for filename in files:
        file_path = os.path.join(sign_images_dir, filename)
        frame = cv2.imread(file_path)
        tasks,index = sd.detect(frame)
        save_path = os.path.join(sign_result_dir, filename)
        draw_res(frame, tasks)
        save_image(frame, save_path)


if __name__ == "__main__":
    os.system("startx")
    time.sleep(1.5)
    # test_front()
    # test_sign_detector()
    # test_cruise()
    test_sign()
    # test_task()
