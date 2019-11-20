# -*- coding: UTF-8 -*-
import sys

sys.path.append('..')
import os
import math
import numpy as np
import random
import time
import matplotlib.pyplot as plt  # 约定俗成的写法plt
import tensorflow as tf
from tqdm import tqdm
import util.mydlib.face_align as align
from util.dlib_face import dlib_face
from util.get_args import get_args
from config import cfg


def log():
    f = open("../logs/20191107_211836_train_log", "r")
    w = open("../logs/20191107_211836_train_log1", "a")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        if 'epoch' in line:
            w.write(line)
            # print(line)

    f.close()
    w.close()


def csv():
    w = open("../data_dir/FG-NET/gt_avg_test.csv", "a")
    lines = []
    lines.append("file_name,apparent_age_avg,apparent_age_std,real_age")
    path = "/media/zouy/workspace/gitcloneroot/age-estimation-pytorch/data_dir/FG-NET/test/"

    # 获取该目录下所有文件，存入列表中
    f = os.listdir(path)
    for name in f:
        split_len = name.split("A").__len__()
        if split_len == 2:
            tmp = name[name.index("A") + 1:name.index(".")]
        else:
            tmp = name[name.index("a") + 1:name.index(".")]
        if tmp[-1].isalpha():
            tmp = tmp[:-1]
        lines.append(name + "," + tmp + ",1," + tmp)
    # 写入csv文件
    for line in lines:
        w.write(line + "\n")
    w.close()


def changename():
    path = "/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/data/dataset/morph2/"
    f = os.listdir(path)
    for name in f:
        oldname = path + "/" + name
        # 设置新文件名
        newname = oldname[0:oldname.index(".")] + ".jpg"

        # 用os模块中的rename方法对文件改名
        os.rename(oldname, newname)


def write_txt():
    path = cfg.dataset.morph2_align + "/"
    train_list = cfg.dataset.morph2_split + "/train_morph2_align.txt"
    # val_list = base + "C3AE_Age_Estimation/data/val_list/val.txt"
    test_list = cfg.dataset.morph2_split + "/test_morph2_align.txt"

    img_list_txt = []
    train_list_txt = []
    # val_list_txt = []
    test_list_txt = []

    f = os.listdir(path)  # 把所有图片名读进来
    # print(f.__len__())#morph2: 52099 morph2_align: 52017

    for name in f:
        # print(name)
        if "M" in name:
            img_list_txt.append(path + name + " " + name[name.index("M") + 1:name.index(".")])
        elif "F" in name:
            img_list_txt.append(path + name + " " + name[name.index("F") + 1:name.index(".")])
    # sort
    # img_list_txt.sort()
    # train_list_txt.sort()

    # random.shuffle(img_list_txt)
    train_list_txt_tmp = img_list_txt[:int(img_list_txt.__len__() * 0.8)]
    # val_list_txt_tmp = img_list_txt[int(img_list_txt.__len__() * 0.7):int(img_list_txt.__len__() * 0.9)]
    test_list_txt_tmp = img_list_txt[int(img_list_txt.__len__() * 0.8):]
    # print(train_list_txt)
    # 计算age_Yn_vector
    for item in tqdm(train_list_txt_tmp):
        train_list_txt_item = item.split(" ")
        age = int(item.split(" ")[1])
        age_vector = np.zeros(11)  # 0-100 101类 11个年龄段
        age_vector[age // 10] = (10 - age % 10) / 10
        age_vector[age // 10 + 1] = age % 10 / 10
        np.insert(age_vector, 0, train_list_txt_item[1])
        train_list_txt.append(list_to_str(train_list_txt_item) + " " + list_to_str(age_vector))
    # for item in val_list_txt_tmp:
    #     val_list_txt_item = item.split(" ")
    #     age = int(item.split(" ")[1])
    #     age_vector = np.zeros(11)
    #     age_vector[age // 10] = (10 - age % 10) / 10
    #     age_vector[age // 10 + 1] = age % 10 / 10
    #     np.insert(age_vector, 0, val_list_txt_item[1])
    #     val_list_txt.append(list_to_str(val_list_txt_item) + " " + list_to_str(age_vector))
    for item in tqdm(test_list_txt_tmp):
        test_list_txt_item = item.split(" ")
        age = int(item.split(" ")[1])
        age_vector = np.zeros(11)
        age_vector[age // 10] = (10 - age % 10) / 10
        age_vector[age // 10 + 1] = age % 10 / 10
        np.insert(age_vector, 0, test_list_txt_item[1])
        test_list_txt.append(list_to_str(test_list_txt_item) + " " + list_to_str(age_vector))

    w1 = open(train_list, "a")
    for line in train_list_txt:
        w1.write(line + "\n")
    w1.close()
    # w2 = open(val_list, "a")
    # for line in val_list_txt:
    #     w2.write(line + "\n")
    # w2.close()
    w3 = open(test_list, "a")
    for line in test_list_txt:
        w3.write(line + "\n")
    w3.close()


def csv_morph2_align():
    # 产生train val test的csv文件比例7:2:1
    path = cfg.dataset.morph2_align + "/"
    train_list = cfg.dataset.morph2_split + "/gt_avg_train.csv"
    val_list = cfg.dataset.morph2_split + "/gt_avg_valid.csv"
    test_list = cfg.dataset.morph2_split + "/gt_avg_test.csv"

    img_list_txt = []
    train_list_txt = []
    val_list_txt = []
    test_list_txt = []
    #设置表头
    train_list_txt.append("file_name,apparent_age_avg,apparent_age_std,real_age")
    val_list_txt.append("file_name,apparent_age_avg,apparent_age_std,real_age")
    test_list_txt.append("file_name,apparent_age_avg,apparent_age_std,real_age")

    f = os.listdir(path)  # 把所有图片名读进来
    # print(f.__len__())#morph2: 52099 morph2_align: 52017

    for name in f:
        # print(name)
        if "M" in name:
            # img_list_txt.append(path + name + " " + name[name.index("M") + 1:name.index(".")])
            img_list_txt.append(name + " " + name[name.index("M") + 1:name.index(".")])
        elif "F" in name:
            # img_list_txt.append(path + name + " " + name[name.index("F") + 1:name.index(".")])
            img_list_txt.append(name + " " + name[name.index("F") + 1:name.index(".")])
    # sort
    # img_list_txt.sort()
    # train_list_txt.sort()

    # random.shuffle(img_list_txt)
    train_list_txt_tmp = img_list_txt[:int(img_list_txt.__len__() * 0.7)]
    val_list_txt_tmp = img_list_txt[int(img_list_txt.__len__() * 0.7):int(img_list_txt.__len__() * 0.9)]
    test_list_txt_tmp = img_list_txt[int(img_list_txt.__len__() * 0.9):]
    # print(train_list_txt)
    # 计算age_Yn_vector
    for item in tqdm(train_list_txt_tmp):
        name = item.split(" ")[0]
        age = item.split(" ")[1]
        train_list_txt.append(name + "," + age + ",1," + age)
    for item in tqdm(val_list_txt_tmp):
        name = item.split(" ")[0]
        age = item.split(" ")[1]
        val_list_txt.append(name + "," + age + ",1," + age)
    for item in tqdm(test_list_txt_tmp):
        name = item.split(" ")[0]
        age = item.split(" ")[1]
        test_list_txt.append(name + "," + age + ",1," + age)

    w1 = open(train_list, "a")
    for line in train_list_txt:
        w1.write(line + "\n")
    w1.close()
    w2 = open(val_list, "a")
    for line in val_list_txt:
        w2.write(line + "\n")
    w2.close()
    w3 = open(test_list, "a")
    for line in test_list_txt:
        w3.write(line + "\n")
    w3.close()


def count_age_group():
    img_list = "/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/data/img_list/img_list.txt"
    f = open(img_list, "r")
    age_group = {"0-9": 0, "10-19": 0, "20-29": 0, "30-39": 0, "40-49": 0, "50-59": 0, "60-69": 0}
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        age = int(line)
        if 0 <= age <= 9:
            age_group["0-9"] += 1
        elif 10 <= age <= 19:
            age_group["10-19"] += 1
        elif 20 <= age <= 29:
            age_group["20-29"] += 1
        elif 30 <= age <= 39:
            age_group["30-39"] += 1
        elif 40 <= age <= 49:
            age_group["40-49"] += 1
        elif 50 <= age <= 59:
            age_group["50-59"] += 1
        elif 60 <= age <= 69:
            age_group["60-69"] += 1
    print(age_group)


def str_to_list(t_str):
    a_list = []
    for c in str(t_str):
        a_list.append(c)
    return a_list


def list_to_str(a_list):
    return " ".join(list(map(str, a_list)))


def plot():
    # 首先定义两个函数（正弦&余弦）
    # X = np.linspace(-np.pi, np.pi, 256, endpoint=True)  # -π to+π的256个值
    # C, S = np.cos(X), np.sin(X)
    # plt.plot(X, C)
    # plt.plot(X, S)
    # # 在ipython的交互环境中需要这句话才能显示出来
    # plt.show()
    base = ""

    x = np.linspace(1, 160, 160, endpoint=True)
    f = open("/media/zouy/workspace/gitcloneroot/C3AE_Age_Estimation/logs/20191107_211836_train_log1", "r")
    lines = f.readlines()
    y = []
    for line in lines:
        line = line.split(" ")
        # print(line)
        y.append(round(float(line[5]), 2))
        f.close()
    y = np.array(y)
    plt.plot(x, y)
    plt.show()


def tf_reduce_mean():
    mean_loss = [1, 2, 3, 5, 6.]
    # mean_loss = tf.cast(mean_loss, tf.float32)
    mean_loss = tf.reduce_mean(mean_loss)

    with tf.Session() as sess:
        print(sess.run(mean_loss))


def lt3_loss():
    f = open("../logs/20191107_152820_train_log1", "r")
    w = open("../logs/20191107_152820_train_log2", "a")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        line_split = line.split(" ")
        if float(line_split[5]) < 3.:
            # print(line_split)
            w.write(line)

    f.close()
    w.close()


def merge_imdb_wiki():
    base = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/C3AE_Age_Estimation/data/dataset/imdb_wiki_crop/"
    f1 = open(base + "imdb_crop_train.txt", "r")
    f2 = open(base + "wiki_crop_train.txt", "r")
    w = open(base + "imdb_wiki_crop_train.txt", "a")
    f1lines = f1.readlines()
    f2lines = f2.readlines()
    w_txt = f1lines + f2lines
    for line in tqdm(w_txt):
        if line is not None and line.__len__() > 1:
            line = line.replace("\n", "")
            w.write(line + "\n")

    f1.close()
    f2.close()
    w.close()


def gen_morph2_align():
    args = get_args()
    start = int(args.start)
    end = int(args.end)
    base = cfg.DATASET
    f = os.listdir(cfg.dataset.morph2)  # 把所有图片名读进来
    f = f[start:end]
    for name in tqdm(f):
        input_path = cfg.dataset.morph2 + "/" + name
        output_path = cfg.dataset.morph2_align + "/" + name
        if os.path.isfile(output_path):  # 已经存在了就跳过
            # print("already exist")
            continue

        align.gen_align_img(input_path, output_path)


def gen_morph2_txt():
    # base = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/"
    base = "/media/zouy/workspace/gitcloneroot/mypython/dataset/"

    f = os.listdir(base + "morph2/")  # 把所有图片名读进来
    w = open(base + "morph2_split/morph2.txt", "a")
    for name in f:
        name = base + "morph2/" + name
        name = name.replace("\n", "")
        w.write(name + "\n")

    w.close()


def dlib_morph2_det():
    path = "/media/zouy/workspace/gitcloneroot/mypython/dataset/morph2_split/morph2.txt"
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    dlib_face(path, 0, lines.__len__())


if __name__ == '__main__':
    csv_morph2_align()
    # x = np.zeros(5)
    # x = np.insert(x, 1, [1, 2])
    # print(x)
    # start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # print(time.strptime(end_time, '%Y-%m-%d %H:%M:%S')-time.strptime(start_time, '%Y-%m-%d %H:%M:%S'))
