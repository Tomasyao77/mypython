# -*- coding: UTF-8 -*-
import sys
sys.path.append("..")
import os
import pandas as pd
from config import cfg
import shutil
from pathlib import Path
from tqdm import tqdm


def log_refine(in_path, out_path):
    f = open(in_path, "r")
    w = open(out_path, "a")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        if 'best val mae' in line:
            w.write(line)
            # print(line)

    f.close()
    w.close()


def fg_net_csv(img_root_path, csv_out_path):
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


if __name__ == "__main__":
    cpfile_fromtxt()
