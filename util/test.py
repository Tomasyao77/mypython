# -*- coding: UTF-8 -*-
import time

dict = {"0": 1, "1": 2, "abc": 'ff'}


def main(dict_=None):

    #print("邮件发送成功-时间: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    mail_msg = "<p>喜报！服务器代码训练完毕！</p><br/>"
    if dict_ is not None:
        for key, value in dict_.items():
            mail_msg += '<p>{key}: {value}</p>'.format(key=key, value=value)
    print(mail_msg)

if __name__ == "__main__":
    main(dict)
