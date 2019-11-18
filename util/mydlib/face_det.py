# encoding:utf-8

import dlib
import numpy as np
import cv2
from tqdm import tqdm
from config import cfg


def rect_to_bb(rect):  # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def detect():
    path = cfg.dataset.morph2 + "/296184_01M20.jpg"
    image_file = path
    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_file)
    # image = resize(image, width=1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        print("fail")
    elif len(rects) == 1:
        print("success")
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)


# 0:未检测到人脸 1:检测到一个人脸
def detect_0_1(path, start=0, end=1):
    # dlib预测器
    detector = dlib.get_frontal_face_detector()

    f = open(path, "r")
    lines = f.readlines()
    f.close()
    lines = lines[start:end]
    detects0_txt = []
    detects1_txt = []
    for file_name in tqdm(lines):
        file_name = file_name.replace("\n", "")
        img = cv2.imread(file_name)
        # print("img/shape:", img.shape)
        detects = detector(img, 1)
        if len(detects) == 0:
            detects0_txt.append(file_name)
        elif len(detects) == 1:
            detects1_txt.append(file_name)
            # print("人脸数：", len(detects))

    w0 = open(path + "_0", "a")
    w1 = open(path + "_1", "a")
    for line in detects0_txt:
        w0.write(line + "\n")
    for line in detects1_txt:
        w1.write(line + "\n")
    w0.close()
    w1.close()


if __name__ == "__main__":
    detect()
