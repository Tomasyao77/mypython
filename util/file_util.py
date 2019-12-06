# -*- coding: UTF-8 -*-
import sys

sys.path.append("..")
import os
import pandas as pd
from config import cfg
import shutil
from pathlib import Path
from tqdm import tqdm
import util.mydlib.face_align as align
from util.get_args import get_args
import numpy as np
import cv2


def log_refine(in_path, out_path):
    f = open(in_path, "r")
    w = open(out_path, "a")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        if 'best val mae:' in line:
            w.write(line)
            # print(line)

    f.close()
    w.close()


def fg_net_nameatoA(img_root_path):
    f = os.listdir(img_root_path)
    for name in tqdm(f):
        oldname = img_root_path + "/" + name
        # 设置新文件名
        if name[3] == "a":
            newname = img_root_path + "/" + name[0:3] + "A" + name[4:]
            # 用os模块中的rename方法对文件改名
            os.rename(oldname, newname)


def fg_net_csv():
    img_root_path = cfg.dataset.fgnet
    csv_out_path = cfg.dataset.fgnet_split
    # 目录不存在先创建
    if not os.path.isdir(str(csv_out_path)):
        os.makedirs(str(csv_out_path))
    w = open(csv_out_path, "a")
    lines = []
    # 表头
    lines.append("file_name,apparent_age_avg,apparent_age_std,real_age")

    f = os.listdir(img_root_path)
    for name in f:
        split_len = name.split("A").__len__()
        # 提取文件名中的年龄
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


# .JPG -> .jpg
def filename_to_lowercase(img_root_path):
    f = os.listdir(img_root_path)
    for name in f:
        oldname = img_root_path + "/" + name
        # 设置新文件名
        newname = oldname[0:oldname.index(".")] + ".jpg"

        # 用os模块中的rename方法对文件改名
        os.rename(oldname, newname)


def list_to_str(a_list):
    return " ".join(list(map(str, a_list)))


# copyfile according to txt
def cpfile_fromtxt():
    img_root = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/age-estimation-pytorch/data_dir/morph2-align/morph2_align"
    txt = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/age-estimation-pytorch/data_dir/morph2-align"
    csv_path = txt + "/gt_avg_test.csv"
    des_path = "/media/gisdom/data11/tomasyao/workspace/pycharm_ws/age-estimation-pytorch/img_dir"
    df = pd.read_csv(str(csv_path))
    # print(df)
    for __, row in tqdm(df.iterrows()):
        img_name = row["file_name"]
        # print(str(i) + ": " + img_name)
        shutil.copyfile(str(Path(img_root).joinpath(img_name)), str(Path(des_path).joinpath(img_name)))


# fgnet leave 1 out 同时生成csv文件
def fg_net_leave1out():
    object_num = 82  # 82个人的各个年龄段的照片共1002张
    origin_dir = cfg.dataset.fgnet
    des_dir_root = cfg.dataset.fgnet_leave1out
    for i in tqdm(range(1, object_num + 1)):
        tmp = str(i) if i > 9 else "0" + str(i)
        des_dir_test = Path(des_dir_root).joinpath(tmp).joinpath("test")
        des_dir_train = Path(des_dir_root).joinpath(tmp).joinpath("train")
        # 目录不存在先创建
        if not os.path.isdir(str(des_dir_train)):
            os.makedirs(str(des_dir_train))
        if not os.path.isdir(str(des_dir_test)):
            os.makedirs(str(des_dir_test))
        # 留一人法: test为一个人的全部图片 train为剩下81个人的全部图片 共82组这样的目录
        f = os.listdir(origin_dir)
        # 使用shutil拷贝图片文件到相应目录
        lines_test = []
        lines_train = []
        # 表头
        lines_test.append("file_name,apparent_age_avg,apparent_age_std,real_age")
        lines_train.append("file_name,apparent_age_avg,apparent_age_std,real_age")
        for name in f:
            # 提取文件名中的年龄
            age = name[name.index("A") + 1:name.index(".")]
            if age[-1].isalpha():
                age = age[:-1]

            if name.startswith("0" + tmp + "A"):  # test dir
                lines_test.append(name + "," + age + ",1," + age)
                if not os.path.isfile(str(Path(des_dir_test).joinpath(name))):
                    shutil.copyfile(str(Path(origin_dir).joinpath(name)), str(Path(des_dir_test).joinpath(name)))
            else:  # train dir
                lines_train.append(name + "," + age + ",1," + age)
                if not os.path.isfile(str(Path(des_dir_train).joinpath(name))):
                    shutil.copyfile(str(Path(origin_dir).joinpath(name)), str(Path(des_dir_train).joinpath(name)))
        # 生成csv文件
        csv_out_path_test = Path(des_dir_root).joinpath(tmp).joinpath("gt_avg_test.csv")
        csv_out_path_train = Path(des_dir_root).joinpath(tmp).joinpath("gt_avg_train.csv")
        # 如果文件已存在则删除 因为后面打开模式是a 表示追加 以免造成数据重复
        if os.path.isfile(csv_out_path_test):
            os.remove(str(csv_out_path_test))
        if os.path.isfile(csv_out_path_test):
            os.remove(str(csv_out_path_train))
        w_test = open(csv_out_path_test, "a")
        w_train = open(csv_out_path_train, "a")
        for line in lines_test:
            line = line.replace("\n", "")
            w_test.write(line + "\n")
        for line in lines_train:
            line = line.replace("\n", "")
            w_train.write(line + "\n")
        w_test.close()
        w_train.close()


# CACD2000
def cacd2000_csv():
    # 产生train val test的csv文件比例7:2:1
    path = cfg.dataset.cacd2000 + "/"
    train_list = cfg.dataset.cacd2000_split + "/gt_avg_train.csv"
    val_list = cfg.dataset.cacd2000_split + "/gt_avg_valid.csv"
    test_list = cfg.dataset.cacd2000_split + "/gt_avg_test.csv"

    train_list_txt = []
    val_list_txt = []
    test_list_txt = []
    # 设置表头
    train_list_txt.append("file_name,apparent_age_avg,apparent_age_std,real_age")
    val_list_txt.append("file_name,apparent_age_avg,apparent_age_std,real_age")
    test_list_txt.append("file_name,apparent_age_avg,apparent_age_std,real_age")

    f = os.listdir(path)  # 把所有图片名读进来
    # print(f.__len__())  # cacd2000: 163446
    # for name in f:
    #     print(name)
    # print(f)

    train_list_txt_tmp = f[:int(f.__len__() * 0.7)]
    val_list_txt_tmp = f[int(f.__len__() * 0.7):int(f.__len__() * 0.9)]
    test_list_txt_tmp = f[int(f.__len__() * 0.9):]
    # 计算age_Yn_vector
    for item in tqdm(train_list_txt_tmp):
        name = item
        age = item.split("_")[0]
        train_list_txt.append(name + "," + age + ",1," + age)
    for item in tqdm(val_list_txt_tmp):
        name = item
        age = item.split("_")[0]
        val_list_txt.append(name + "," + age + ",1," + age)
    for item in tqdm(test_list_txt_tmp):
        name = item
        age = item.split("_")[0]
        test_list_txt.append(name + "," + age + ",1," + age)

    # 如果文件已存在则删除 因为后面打开模式是a 表示追加 以免造成数据重复
    if os.path.isfile(train_list):
        os.remove(str(train_list))
    if os.path.isfile(val_list):
        os.remove(str(val_list))
    if os.path.isfile(test_list):
        os.remove(str(test_list))

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


# fgnet align
def fg_net_align():
    args = get_args()
    start = int(args.start)
    end = int(args.end)
    f = os.listdir(cfg.dataset.fgnet)  # 把所有图片名读进来
    f = f[start:end]
    # 目录不存在先创建
    if not os.path.isdir(cfg.dataset.fgnet_align):
        os.makedirs(cfg.dataset.fgnet_align)

    for name in tqdm(f):
        input_path = cfg.dataset.fgnet + "/" + name
        output_path = cfg.dataset.fgnet_align + "/" + name
        if os.path.isfile(output_path):  # 已经存在了就跳过
            continue

        align.gen_align_img(input_path, output_path)


if __name__ == "__main__":
    # cpfile_fromtxt()
    # fg_net_nameatoA("/media/zouy/workspace/gitcloneroot/mypython/dataset/FG-NET")
    # fg_net_leave1out()

    # f = open("/media/zouy/workspace/gitcloneroot/mypython/logs/20191122_094322_train_log1", "r")
    # lines = f.readlines()
    # arr = []
    # for line in lines:
    #     best_val_mae = line.replace("\n", "").split(" ")[-1]
    #     arr.append(float(best_val_mae))
    # print(arr)

    # fg_net_leave1out()

    # random copy
    # data_dir = cfg.dataset.morph2
    # f = os.listdir(data_dir)  # 把所有图片名读进来
    # # print(f.__len__())  # cacd2000: 163446
    # np.random.shuffle(f)
    # pick = f[:20]
    # des_dir = "/home/zouy/Desktop/mdpi_age/img/dataset/morph2"
    # for name in pick:
    #     shutil.copyfile(data_dir + "/" + name, des_dir + "/" + name)
    #
    # #cv2 resize
    # data_dir = "/home/zouy/Desktop/mdpi_age/img/dataset/morph2"
    # f = os.listdir(data_dir)
    # for name in f:
    #     img = cv2.imread(data_dir + "/" + name)
    #     img_resize = cv2.resize(img, (160, 210))
    #     cv2.imwrite(data_dir + "/" + name, img_resize)

    # fgnet
    # data_dir = "/home/zouy/Desktop/mdpi_age/img/dataset/FG-NET"
    # f = os.listdir(data_dir)
    # for name in f:
    #     img = cv2.imread(data_dir + "/" + name)
    #     img_resize = cv2.resize(img, (160, 200))
    #     cv2.imwrite(data_dir + "/" + name, img_resize)

    # dlib data
    # data_dir = cfg.dataset.morph2
    # f = os.listdir(data_dir)  # 把所有图片名读进来
    # # print(f.__len__())  # cacd2000: 163446
    # np.random.shuffle(f)
    # pick = f[:20]
    # des_dir = "/home/zouy/Desktop/mdpi_age/img/dlib"
    # for name in pick:
    #     shutil.copyfile(data_dir + "/" + name, des_dir + "/" + name)

    # dlib crop
    base = "/home/zouy/Desktop/mdpi_age/img/dlib"
    img1 = base + "/241301_05M25.jpg"
    img2 = base + "/316052_00M17.jpg"
    align.demo(img1, base + "/241301_05M25_dect.jpg", base + "/241301_05M25_adjust.jpg")
    align.demo(img2, base + "/316052_00M17_dect.jpg", base + "/316052_00M17_adjust.jpg")
