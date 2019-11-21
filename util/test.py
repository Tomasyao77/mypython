# -*- coding: UTF-8 -*-
import time
# import sys
# sys.path.append(".")
# sys.path.append("..")
# print(sys.path)
import util.file_util
import temp_py
import numpy as np

dict = {"0": 1, "1": 2, "abc": 'ff'}


def main(dict_=None):
    # print("邮件发送成功-时间: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # mail_msg = "<p>喜报！服务器代码训练完毕！</p><br/>"
    # if dict_ is not None:
    #     for key, value in dict_.items():
    #         mail_msg += '<p>{key}: {value}</p>'.format(key=key, value=value)
    # print(mail_msg)

    x = [1, 2, 3]
    print(x[1:5])


if __name__ == "__main__":
    # main(dict)
    # 判断各个数据集是否有重复元素，经检查的确没有
    # f1 = open("/media/zouy/workspace/gitcloneroot/mypython/dataset/morph2_split/gt_avg_train.csv", "r")
    # f2 = open("/media/zouy/workspace/gitcloneroot/mypython/dataset/morph2_split/gt_avg_valid.csv", "r")
    # f3 = open("/media/zouy/workspace/gitcloneroot/mypython/dataset/morph2_split/gt_avg_test.csv", "r")
    # list1 = f1.readlines()
    # list2 = f2.readlines()
    # list3 = f3.readlines()
    # c = [x for x in list1 if x in list2]
    # d = [x for x in list1 if x in list3]
    # e = [x for x in list2 if x in list3]
    # print (c)
    # print (d)
    # print (e)

    # 求平均值
    # val_mae_list = [1, 3, 4, 6, 7]
    # print(np.array(val_mae_list).mean())

    #矩阵相乘
    # import torch
    # import torch.nn.functional as F
    #
    # preds = []
    # outputs = torch.Tensor([[1.,2.,3.]])
    # preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
    # outputs = torch.Tensor([[2., 2., 3.]])
    # preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
    # preds = np.concatenate(preds, axis=0)  # [1 2 3 1 2 3 1 2 3]
    # print(preds)
    # ages = np.arange(0,3)
    # ave_preds = (preds * ages).sum(axis=-1)
    #
    # print(f"ave_preds:{ave_preds}")

