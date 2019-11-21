# -*- coding: UTF-8 -*-
import time
import sys
sys.path.append("..")
# print(sys.path)
import util.file_util
import temp_py
import numpy as np
from util.myplot import plot

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

    #fgnet leave1out 82group mean
    fgnet = [2.324496947471157, 3.884720985786375, 8.176459764953826, 7.836859238620348, 9.650200213792024, 14.764931480953308,
     4.4528422183824246, 3.1248625078306325, 2.1002871192359382, 1.3584724781078084, 1.6585260410481995,
     4.67326149402399, 3.771500607513842, 3.109655131950633, 0.8696305520685749, 1.378411414731276, 4.4422052571353365,
     3.2281120103383603, 3.9661514938751736, 4.157233624738224, 2.1179430965000443, 2.3032440626304718,
     2.2781196680721156, 2.6216442903290127, 3.2901473796350533, 0.9555723975440039, 2.556125027790521,
     3.3639174159625926, 2.1677573417950766, 2.829885177519235, 1.496235461421784, 3.0430500859885967,
     1.5436541292620125, 2.928312363341053, 1.3157583378458841, 1.2035647587343747, 1.428000067144735,
     2.522155618893979, 5.70952551204684, 1.2626470768082474, 1.9480038638078612, 1.396584069308792, 1.5411064185307337,
     1.805358252725238, 3.051455590666264, 1.3542098391156445, 4.728647423288164, 5.097944150799565, 1.6980498664733872,
     2.1173985850446555, 1.2963189759902798, 2.122171926042547, 1.9528249598045113, 2.2821327289175795,
     1.183591312283402, 1.4659432084406194, 2.564241746350115, 1.3227850864504687, 1.9813655820581237,
     2.060695666236178, 5.176551806059805, 5.08703499026445, 3.262304645556486, 0.9466801766681728, 1.1759147747203242,
     1.0571091818834706, 4.5765181793002965, 1.5306398142716326, 0.990286774236338, 0.9983108613961221,
     2.820391794964076, 4.097985392696299, 1.1918933414668629, 0.9289132297891916, 0.7174557348421577,
     0.9781572521514809, 0.8384572496202398, 0.8899930072765874, 0.9752833850183364, 0.8967771430578199,
     0.9044048714191605, 1.5474048090722838]
    print(np.array(fgnet).mean())#2.688114359998644
    x = np.linspace(1, 82, 82, endpoint=True)
    plot(x, fgnet, )