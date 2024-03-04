# 从互联网获取数据

> 原文：<https://www.pythonforbeginners.com/python-on-the-web/fetching-data-from-the-internet>

### 什么是 Urllib2？

urllib2 是一个用于获取 URL 的 Python 模块。

### 我能做什么

它以 urlopen 函数的形式提供了一个非常简单的接口。

Urlopen 能够使用各种不同的协议获取 URL，比如
(http、ftp、file)。

它还提供了一个处理基本认证、cookies、代理
等的接口。

这些是由称为处理程序和打开程序的对象提供的。

### HTTP 请求

HTTP 基于请求和响应，客户端发出请求，服务器发送响应。

这个响应是一个类似文件的对象，这意味着您可以调用。阅读()
上的回复。

### 怎么用？

```py
import urllib2
response = urllib2.urlopen('http://python.org/')
html = response.read() 
```

### 用户代理

您也可以使用 urllib2 添加自己的头。

有些网站不喜欢被程序浏览。

默认情况下，urllib2 将自己标识为 Python-urllib/x.y(其中 x 和 y 是 Python 版本的主版本号和次版本号，
，这可能会使站点混淆，或者根本不起作用。

浏览器通过用户代理头标识自己。

参见我们的 [Urllib2 用户代理](https://www.pythonforbeginners.com/code-snippets-source-code/python-modules-urllib2-user-agent)帖子，它描述了如何在程序中使用它。

### 获取 HTTP 头

让我们写一个小脚本，从一个网站获取 HTTP 头。

```py
import urllib2
response = urllib2.urlopen("http://www.python.org")
print "-" * 20
print "URL : ", response.geturl()

headers = response.info()
print "-" * 20
print "This prints the header: ", headers
print "-" * 20
print "Date :", headers['date']
print "-" * 20
print "Server Name: ", headers['Server']
print "-" * 20
print "Last-Modified: ", headers['Last-Modified']
print "-" * 20
print "ETag: ", headers['ETag']
print "-" * 20
print "Content-Length: ", headers['Content-Length']
print "-" * 20
print "Connection: ", headers['Connection']
print "-" * 20
print "Content-Type: ", headers['Content-Type']
print "-" * 20 
```

##### 将给出类似如下的输出:

———————
网址:http://www.python.org
—————
这是打印的标题:日期:Fri，2012 年 10 月 12 日 08:09:40 GMT
服务器:Apache/2.2.16 (Debian)
最后修改时间:周四，2012 年 10 月 11 日 22:36:55 GMT
ETag:" 105800d-4de 0-4c BD 035514 fc0 "【内容

———
日期:Fri，2012 年 10 月 12 日 08:09:40 GMT
————
服务器名:Apache/2 . 2 . 16(Debian)
————
最后修改时间:2012 年 10 月 11 日星期四 22:36:55 GMT
—————
ETag:" 105800d-4de 0-4c bdg