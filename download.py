# -*- coding: UTF-8 -*-
import os
import urllib2
import sys
import re

#嫦娥人脸图片
def download():
    # txt文件所在路径
    with open('/media/zouy/download/mysql_txt/2018-11-22_1625_face.txt', 'r') as file:
        urls = file.readlines()
        for index, item in enumerate(urls):
            # 0(did) 1(dpid) 2(photoid) 3(roughness_score) 4(url)
            attr = item.split("\t")
            img_url = attr[4]
            file_path = '/media/zouy/download/mysql_txt/2018-11-22_1625_face'
            file_name = attr[0] + '_' + attr[1] + '_' + attr[2] + '_' + attr[3]

            try:
                # 是否有这个路径
                if not os.path.exists(file_path):
                    # 创建路径
                    os.makedirs(file_path)
                # 获得图片后缀
                file_suffix = os.path.splitext(img_url)[1]
                #print("file_suffix: " + file_suffix)
                # 拼接图片名（包含路径）
                filename = '{}{}{}{}'.format(file_path, os.sep, file_name, file_suffix)
                #print("filename: " + filename)
                # 下载图片，并保存到文件夹中
                f = urllib2.urlopen(img_url).read()
                open(filename.strip(), 'wb').write(f)

            except IOError as e:
                print("IOError")
            except Exception as e:
                print(e)
            if index % 10 == 0:
                print(index)

def onlyImg():
    # txt文件所在路径
    with open('/media/zouy/download/mysql_txt/2018-11-22_1625_face.txt', 'r') as file:
        urls = file.readlines()
        for index, item in enumerate(urls):
            # 0(did) 1(dpid) 2(photoid) 3(roughness_score) 4(url)
            attr = item.split("\t")
            length = len(urls)
            img_url = attr[4]
            print(img_url.strip())

            try:
                # 仅保存imgurl
                f = open('/media/zouy/download/mysql_txt/2018-11-22_1625_face_imgurl.txt', 'a')
                f.write(img_url)

            except IOError as e:
                print("IOError")
            except Exception as e:
                print(e)

def rename():
    fileList = os.listdir("/media/zouy/download/mysql_txt/2018-11-22_1625_face")  # 待修改文件夹
    currentpath = os.getcwd()  # 得到进程当前工作目录
    os.chdir("/media/zouy/download/mysql_txt/2018-11-22_1625_face")  # 将当前工作目录修改为待修改文件夹的位置

    for fileName in fileList:  # 遍历文件夹中所有文件
        os.rename(fileName, fileName.strip())  # 文件重新命名
    os.chdir(currentpath)  # 改回程序运行前的工作目录
    sys.stdin.flush()  # 刷新


if __name__ == '__main__':
    download()
    # #onlyImg()
    # rename()

