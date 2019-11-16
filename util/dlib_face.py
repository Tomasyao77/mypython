import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2  # 图像处理的库OpenCv
import os
import threading
import math
from tqdm import tqdm


# my custom module
# from myplot import plot


def dlib_face(filepath, start=0, end=1):
    # dlib预测器
    detector = dlib.get_frontal_face_detector()

    path = filepath
    f = open(path, "r")
    lines = f.readlines()
    lines = lines[start:end]
    thread_name = threading.current_thread().getName()
    detects0_txt = []
    detects1_txt = []
    for file_name in tqdm(lines, desc=thread_name):
        file_name = file_name.replace("\n", "")
        img = cv2.imread(file_name)
        # print("img/shape:", img.shape)
        # dlib检测
        detects = detector(img, 1)
        if len(detects) == 0:
            detects0_txt.append(file_name)
        elif len(detects) == 1:
            detects1_txt.append(file_name)
            # print("人脸数：", len(detects))

    w0 = open(path + "_0_" + thread_name, "a")
    w1 = open(path + "_1_" + thread_name, "a")
    for line in detects0_txt:
        w0.write(line + "\n")
    for line in detects1_txt:
        w1.write(line + "\n")
    w0.close()
    w1.close()
    f.close()

    # print("总图片数:" + str(total))  # wiki_crop:62328(处理前)
    # plot(x, y, dict={"type": "bar", "xlabel": "dir_no", "ylabel": "count"})

    # 记录人脸矩阵大小
    # height_max = 0
    # width_sum = 0

    # # 计算要生成的图像img_blank大小
    # for k, d in enumerate(dets):
    #     # 计算矩形大小
    #     # (x,y), (宽度width, 高度height)
    #     pos_start = tuple([d.left(), d.top()])
    #     pos_end = tuple([d.right(), d.bottom()])
    #
    #     # 计算矩形框大小
    #     height = d.bottom() - d.top()
    #     width = d.right() - d.left()
    #
    #     # 处理宽度
    #     width_sum += width
    #     # 处理高度
    #     if height > height_max:
    #         height_max = height
    #     else:
    #         height_max = height_max
    #
    # # 绘制用来显示人脸的图像的大小
    # # print("img_blank的大小：")
    # # print("高度", height_max, "宽度", width_sum)
    #
    # # 生成用来显示的图像
    # img_blank = np.zeros((height_max, width_sum, 3), np.uint8)
    # print(img_blank.shape)
    #
    # # 记录每次开始写入人脸像素的宽度位置
    # blank_start = 0
    #
    # # 将人脸填充到img_blank
    # for k, d in enumerate(dets):
    #
    #     height = d.bottom() - d.top()
    #     width = d.right() - d.left()
    #
    #     # 填充
    #     for i in range(height):
    #         for j in range(width):
    #             img_blank[i][blank_start + j] = img[d.top() + i][d.left() + j]
    #     # 调整图像
    #     blank_start += width
    #
    # cv2.namedWindow("img_faces", 2)
    # cv2.imshow("img_faces", img_blank)
    # cv2.waitKey(0)

    # resize to 128*128 and then save to filepath
    # res = cv2.resize(img_blank, (128, 128))
    # cv2.imwrite(path + file_name[0:file_name.index(".")] + "_128*128.jpg", res)


def gen_imglist():
    path = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop/"
    dir_count = 100
    w = open(path + "imdb_crop_list", "a")
    w_arr = []
    for i in range(dir_count):
        if i < 10:
            mid_dir = "0" + str(i)
        else:
            mid_dir = str(i)

        img_path_root = path + mid_dir
        # 获取该目录下所有文件，存入列表中
        f = os.listdir(img_path_root)

        for file_name in f:
            img_path = img_path_root + "/" + file_name
            w_arr.append(img_path)

    for line in w_arr:
        w.write(line + "\n")
    w.close()


def start_dlib_face():
    threading_num = 1
    threading_pool = []
    path = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop/imdb_crop_list"
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    gap = math.ceil(lines.__len__() / threading_num)
    print(lines.__len__())
    print(gap)

    start = 0
    end = gap
    for i in range(threading_num):
        tmp_thread = threading.Thread(target=dlib_face, args=(path, start, end), name="thread" + str(int(i)))
        threading_pool.append(tmp_thread)
        start = end
        end += gap

    # 启动线程
    for t_pool in threading_pool:
        t_pool.start()

    # 等待线程中止
    for t_pool in threading_pool:
        t_pool.join()


def remove_n():
    f = open("/media/zouy/workspace/gitcloneroot/mypython/dataset/wiki_crop_dlibdetect/wiki_crop_list_1_thread0", "r")
    w = open("/media/zouy/workspace/gitcloneroot/mypython/dataset/wiki_crop_dlibdetect/wiki_crop_list_1", "a")
    lines = f.readlines()
    arr = []
    for line in lines:
        if line.__len__() > 1:
            line = line.replace("\n", "")
            w.write(line + "\n")

    f.close()
    w.close()


def gen_txt():
    f = open(
        "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop_dlibdetect/imdb_crop_list_1_thread0",
        "r")
    w = open(
        "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop_dlibdetect/imdb_crop_train.txt",
        "a")
    lines = f.readlines()
    w_arr = []
    for line in tqdm(lines):
        # /media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/wiki_crop/00/18763900_1943-01-23_2008.jpg
        # /media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop/99/nm0005499_rm1748736000_1939-9-1_1999.jpg
        line = line.replace("\n", "")
        imgpath = line
        line = line[line.rindex("/") + 1:line.rindex(".")]
        line = line.split("_")
        start = line[-2].split("-")[0]  # 起始日期
        end = line[-1]
        age = int(end) - int(start)
        age_vector = np.zeros(12)
        if 0 <= age < 110:  # 不能越界
            age_vector[age // 10] = (10 - age % 10) / 10
            age_vector[age // 10 + 1] = age % 10 / 10
            w_arr.append(imgpath + " " + str(age) + " " + list_to_str(age_vector))

    for line in w_arr:
        # WIKI 38893有效图片数量
        # IMDB 324963有效图片数量
        w.write(line + "\n")

    f.close()
    w.close()


def list_to_str(a_list):
    return " ".join(list(map(str, a_list)))


def split_imdb_train():
    f = open("/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop_dlibdetect/imdb_crop_train.txt","r")
    wa = open("/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop_dlibdetect/imdb_crop_train_a.txt","a")
    wb = open("/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop_dlibdetect/imdb_crop_train_b.txt","a")
    wc = open("/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython/dataset/imdb_crop_dlibdetect/imdb_crop_train_c.txt","a")
    lines = f.readlines()
    gap = int(lines.__len__() / 3)
    wa_txt = lines[0:gap]
    wb_txt = lines[gap:gap*2]
    wc_txt = lines[gap*2:]
    for line in wa_txt:
        line = line.replace("\n", "")
        wa.write(line + "\n")
    for line in wb_txt:
        line = line.replace("\n", "")
        wb.write(line + "\n")
    for line in wc_txt:
        line = line.replace("\n", "")
        wc.write(line + "\n")

    f.close()
    wa.close()
    wb.close()
    wc.close()

if __name__ == "__main__":
    split_imdb_train()
