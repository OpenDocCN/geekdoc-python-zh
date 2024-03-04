# Python 中的套接字示例

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/python-socket-examples>

Python 中的 socket 模块提供了对 BSD 套接字接口的访问。

这篇文章将展示如何使用套接字函数的例子。

要阅读更多关于 socket 模块及其功能的内容，我推荐官方文档[这里](https://docs.python.org/library/socket.html "Sockets-Module")。

### 获取 fqdn(完全限定域名)

```py
 print socket.getfqdn("8.8.8.8")
>>google-public-dns-a.google.com 
```

### 将主机名转换为 IPv4 地址格式

```py
 print socket.gethostbyname("www.python.org")
>>82.94.164.162 
```

### 将主机名转换为 IPv4 地址格式，扩展接口

```py
 print socket.gethostbyname_ex("www.python.org")
>>('python.org', [], ['82.94.164.162']) 
```

### 返回机器的主机名..

```py
 print socket.gethostname()
>>Virtualbox123 
```

### 脚本

```py
 The script below, will return the IP address of the hostname given. 

If it cannot find the hostname, it will print the error to screen using 
the exception handling "gaierror" (raised for address-related errors). 
```

```py
import socket
name = "wwaxww.python.org"
try:
    host = socket.gethostbyname(name)
    print host
except socket.gaierror, err:
    print "cannot resolve hostname: ", name, err

```

```py
 If you look in the name, I misspelled the address to let you see how the error
message would look like

>>cannot resolve hostname:  wwaxww.python.org [Errno -5] No address associated
with hostname

And if I run it again with the correct spelling 
```

```py
import socket
name = "www.python.org"
try:
    host = socket.gethostbyname(name)
    print host
except socket.gaierror, err:
    print "cannot resolve hostname: ", name, err

```

```py
 >>82.94.164.162 
```