# -*- coding: UTF-8 -*-
import time
import sys
import torch

sys.path.append("..")
# print(sys.path)
import util.file_util
import temp_py
import numpy as np
from util.myplot import plot
import torch
import torch.nn as nn
from collections import OrderedDict
from collections import namedtuple, defaultdict

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

    # 矩阵相乘
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

    # fgnet leave1out 82group mean
    # fgnet = [4.558679055698504, 4.089232703374215, 15.753263838436197, 10.9387005638554, 11.08894839653438,
    #          14.633448810729869, 6.562932037542684, 2.847672643440604, 2.8886580986430577, 1.5099549655850482,
    #          3.5230701631104986, 5.0372386709851185, 3.4695946297675158, 6.69201779221204, 2.358184934450772,
    #          2.393789819664985, 5.183084822960762, 3.5691337732468082, 4.851172768530266, 3.364469281452899,
    #          2.7844553667615863, 4.116750135728902, 3.897367501434147, 3.6143821666530935, 4.212606655664039,
    #          1.5057584495560787, 6.9657248010034065, 6.857343249834019, 5.010662301926119, 4.016719132756094,
    #          2.0282684983605574, 6.8082838567619985, 2.9035493004527684, 4.393907617255066, 3.2018666205864994,
    #          1.0036570737792814, 2.8840504871918227, 3.092247195338667, 8.146335784484775, 2.1103913326798973,
    #          2.371286019277005, 2.2456423855687575, 2.183512727159518, 1.6771903864919984, 3.3264837590525262,
    #          1.684567784511387, 6.616519269094604, 8.641984597786351, 2.8880702922365593, 2.66350693895461,
    #          3.4851982200346403, 2.9289607020194635, 1.765400778059939, 4.60603623750983, 1.940135904036342,
    #          1.1275667997802572, 3.144880322507926, 4.5122116966121055, 2.8265195609633214, 4.388634244042032,
    #          5.4016810457672495, 10.996691764068139, 4.197130482657569, 1.6799665465501843, 1.3216434840819293,
    #          2.532278216992817, 5.504362138443799, 3.551673756840686, 1.1567874551437536, 2.427693408960324,
    #          4.342752241371624, 3.5094254063795915, 3.3839370606688908, 2.427353181612691, 2.06658371334074,
    #          3.2313890692636047, 2.446070902368911, 1.8281002564270268, 2.2727996981190173, 1.1248018006673004,
    #          1.0572577440254147, 3.582392105859401]
    # print(np.array(fgnet).mean())
    # # 2.688114359998644
    # # 4.023593358582199
    # x = np.linspace(1, 82, 82, endpoint=True)
    # plot(x, fgnet, )

    # fgnet_align leave1out 82group mean
    # fgnet = [4.6036425716315685, 3.7418737606745736, 16.15002073807659, 10.997760033020596, 12.268097944632405,
    #          14.424629302054376, 5.555029940410923, 4.406085180012157, 2.5331660687290003, 2.312366495716422,
    #          3.321245373852932, 4.626343896074824, 4.051689311784171, 3.4819463860557938, 2.678153013759913,
    #          2.4652825420097306, 4.173336380576239, 4.3295904483305545, 5.39946058318656, 4.202175948036216,
    #          3.321490332261863, 2.602163139236023, 3.763842615875443, 2.6343561643690228, 3.9465138123014754,
    #          1.6076479591571085, 5.606039930153309, 7.466402292620529, 3.5932467551442566, 3.88382229590985,
    #          2.422059861572157, 5.756819861716004, 1.974291824049983, 4.430964438147407, 2.0372085660279042,
    #          1.1622081382382385, 2.104953933561792, 2.0355821428284093, 7.166350121859471, 4.450502336305427,
    #          1.7175203418128162, 1.9708416184918014, 4.124711691210021, 2.341617188616623, 2.4079623726466495,
    #          1.378663998116532, 5.984039603677884, 6.996912438980796, 2.830609345671085, 3.1241154160361013,
    #          4.926517340762971, 4.484591669717494, 2.456429351008934, 2.334232079966365, 2.5095198202974345,
    #          1.1062630559045683, 3.304635112054229, 2.5767483346280917, 2.507683119487479, 2.261131264671998,
    #          3.5113220726547856, 9.155385778246169, 4.104482853787226, 0.4019951801530704, 1.5531300425339254,
    #          1.1968205214751462, 5.174041369150943, 3.4662670414175367, 2.1716993500237254, 1.8552699784028683,
    #          5.499923086913788, 5.323143827781214, 1.9708509763378108, 2.039813610680041, 0.9235127331809178,
    #          1.310434860208733, 1.7616490980471038, 3.241296440983052, 1.5864755743020151, 1.5770451886468704,
    #          0.9894062614556681, 2.765649648993292]
    # print(np.array(fgnet).mean())
    # # 2.6764195294146678
    # # 3.7879600377938654
    # x = np.linspace(1, 82, 82, endpoint=True)
    # plot(x, fgnet, )

    # batch_size = 128
    # nrow = round((2 * batch_size) ** 0.5)
    # print(nrow)

    # z = torch.tensor([1.,2.,3.,4.,5.,-3.])
    # z_l = torch.rand_like(z)
    # print(z_l)

    # dims = (100, 64, 64 // 2, 64 // 4)
    # for (in_dim, out_dim) in zip(dims[:-1], dims[1:]):
    #     print(in_dim, out_dim)
        #100 64 32
        #64 32 16

    # out = torch.tensor([[1,2],[3,2]])
    # out = out.flatten(1, -1)
    # print(out.flatten(1, -1))

    # t = "_".join([])
    # print(t)

    # losses = defaultdict(lambda: [])
    # losses["valid"].append(1.2)
    # losses["valid"].append(3.1)
    # losses["valid"].append(2.3)
    # print(np.array(losses["valid"]).mean())

    NUM_AGES = 10
    NUM_GENDERS = 2

    gender = 1
    gender_tensor = -torch.ones(NUM_GENDERS)
    gender_tensor[int(gender)] *= -1 #[1, 2]
    # repeat(10, 5)意思是每个维度相应变成以往的多少倍
    # 1*10=10 2*5=10 -> (10,10)
    gender_tensor = gender_tensor.repeat(NUM_AGES,
                                         NUM_AGES // NUM_GENDERS)  # apply gender on all images
    age_tensor = -torch.ones(NUM_AGES, NUM_AGES)

    for i in range(NUM_AGES):
        age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

    l = torch.cat((age_tensor, gender_tensor), 1)
    print(l)


