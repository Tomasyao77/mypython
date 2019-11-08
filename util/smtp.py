# -*- coding: UTF-8 -*-
import smtplib
import sys
import time
from datetime import datetime

from email.mime.text import MIMEText
from email.header import Header


# from email.mime.multipart import MIMEMultipart #发送附件
# from email.mime.image import MIMEImage #在html中添加图片

def main(dict_=None):
    # print '参数个数为:', len(sys.argv), '个参数'
    # print '参数列表:', str(sys.argv)
    # print '脚本名为：', sys.argv[0]
    # for i in range(1, len(sys.argv)):
    #     print '参数 %s 为：%s' % (i, sys.argv[i])

    # 第三方 SMTP 服务
    mail_host = "smtp.qq.com"  # 设置服务器
    mail_user = "772052352@qq.com"  # 用户名
    mail_pass = "deeyykzpzqqdbbdb"  # 口令(pop3/smtp授权码)

    sender = '772052352@qq.com'
    receivers = ['772052352@qq.com', 'zouyao@whut.edu.cn']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    # message = MIMEText('Python 邮件发送测试...', 'plain', 'utf-8')
    mail_msg = "<p>喜报！服务器代码训练完毕！</p>"
    if dict_ is not None:
        for key, value in dict_.items():
            mail_msg += '<p>{key}: {value}</p>'.format(key=key, value=value)
    #print(mail_msg)
    message = MIMEText(mail_msg, 'html', 'utf-8')
    message['From'] = Header("tomasyao", 'utf-8')
    message['To'] = Header("772052352@qq.com", 'utf-8')

    subject = 'Python SMTP.'
    message['Subject'] = Header(subject, 'utf-8')  # 邮件标题

    try:
        smtpObj = smtplib.SMTP_SSL()  # 这里不能用smtplib.SMTP()否则系统假死
        smtpObj.connect(mail_host, 465)  # 465 为 qq SMTP 端口号
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功-时间: " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    except smtplib.SMTPException:
        print("Error: 无法发送邮件")

def date_gap(start, end):
    a = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    b = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    t1 = time.mktime(a.timetuple()) * 1000 + a.microsecond / 1000
    t2 = time.mktime(b.timetuple()) * 1000 + b.microsecond / 1000
    a = t2 - t1
    b = round(a / 1000 / 60, 2) #min 保留两位小数

    return str(b) + "min"

# if __name__ == "__main__":
    # main(dict_ = {"0": 1, "1": 2, "abc": 'ff'})
    # print(date_gap())
    # print(13923//4 * 5)