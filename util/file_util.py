# -*- coding: UTF-8 -*-
import os


def log():
    f = open("../log/log_t", "r")
    w = open("../log/log_t1", "a")
    lines = f.readlines()  # 读取全部内容
    for line in lines:
        if 'best val mae' in line:
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
    path = "/media/zouy/workspace/gitcloneroot/age-estimation-pytorch/data_dir/FG-NET/test"
    f = os.listdir(path)
    for name in f:
        oldname = path + "/" + name
        # 设置新文件名
        newname = oldname[0:oldname.index(".")] + ".jpg"

        # 用os模块中的rename方法对文件改名
        os.rename(oldname, newname)


if __name__ == '__main__':
    csv()
