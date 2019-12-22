# encoding:utf-8

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from config import cfg


def rect_to_bb(rect, height=0, width=0):  # 获得人脸矩形的坐标信息
    #要注意处理越界!!!
    x = 0 if rect.left() < 0 else rect.left()
    y = 0 if rect.top() < 0 else rect.top()

    right = width if rect.right() > width else rect.right()
    w = right - x
    bottom = height if rect.bottom() > height else rect.bottom()
    h = bottom - y

    return (x, y, w, h)


def face_alignment(faces):
    path = cfg.dlib68dat
    predictor = dlib.shape_predictor(path)  # 用来预测关键点
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)  # 注意输入的必须是uint8类型
        order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
            cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,  # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)  # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned


def demo(img_path, dect_path, adjust_path):
    path = img_path
    im_raw = cv2.imread(path).astype('uint8')

    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    rects = detector(gray, 1)
    print(rects)

    src_faces = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect, height, width)
        detect_face = im_raw[y:y + h, x:x + w]
        src_faces.append(detect_face)
        print(f"x:{x} y:{y} w:{w} h:{h}")
        cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(dect_path, im_raw)
    faces_aligned = face_alignment(src_faces)

    # cv2.imshow("src", im_raw)
    i = 0
    for face in faces_aligned:
        cv2.imwrite(adjust_path, face)
        # cv2.imshow("det_{}".format(i), face)
        i = i + 1
    # cv2.waitKey(0)


def gen_align_img(img_path, out_path=None):
    im_raw = cv2.imread(img_path).astype('uint8')
    # 人脸检测
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    rects = detector(gray, 1)
    # 画矩形 放文字
    src_faces = []
    # try:
    if rects.__len__() >= 1:
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = rect_to_bb(rect, height, width)
            detect_face = im_raw[y:y + h, x:x + w]
            src_faces.append(detect_face)
            # cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
            #             1)
        # cv2.imshow("src", im_raw)
        # 人脸对齐
        faces_aligned = face_alignment(src_faces)
        if faces_aligned.__len__() == 1:
            print("align success")
            # cv2.imshow("det", faces_aligned[0])
            if out_path is not None:
                print(f"start write:{out_path}")
                cv2.imwrite(out_path, faces_aligned[0])
    else:
        print("false:" + img_path)
        # except:
        #     print("except:"+img_path)


if __name__ == "__main__":
    demo()
    # gen_align_img(cfg.dataset.morph2 + "/052791_0M46.jpg", cfg.dataset.morph2_align + "/052791_0M46.jpg")
