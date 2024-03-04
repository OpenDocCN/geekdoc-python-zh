# Python 中的端口扫描器

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/port-scanner-in-python>

## 概观

这篇文章将展示如何用 Python 编写一个小巧易用的端口扫描程序。使用 Python 有很多方法可以做到这一点，我将使用内置的模块套接字来实现。

## 套接字

Python 中的 socket 模块提供了对 BSD 套接字接口的访问。它包括用于处理实际数据通道的 socket 类，以及用于网络相关任务的函数，例如将服务器名称转换为地址和格式化要通过网络发送的数据。

套接字在互联网上被广泛使用，因为它们是你的计算机所进行的任何一种网络通信的基础。

INET 套接字至少占正在使用的套接字的 99%。您使用的 web 浏览器打开一个套接字并连接到 web 服务器。

任何网络通信都要通过套接字。

有关插座模块的更多信息，请参见[官方](https://docs.python.org/2/howto/sockets.html "socket_python")文档。

## 套接字函数

在开始我们的示例程序之前，让我们看看我们将要使用的一些套接字函数。

```py
 Syntax for creating a socket
sock = socket.socket (socket_family, socket_type)

Creates a stream socket
sock = socket.socket (socket.AF_INET, socket.SOCK_STREAM)

AF_INET 
Socket Family (here Address Family version 4 or IPv4) 

SOCK_STREAM Socket type TCP connections 

SOCK_DGRAM Socket type UDP connections 

Translate a host name to IPv4 address format 
gethostbyname("host") 

Translate a host name to IPv4 address format, extended interface
socket.gethostbyname_ex("host")  

Get the fqdn (fully qualified domain name)
socket.getfqdn("8.8.8.8")  

Returns the hostname of the machine..
socket.gethostname()  

Exception handling
socket.error
```

## 使用 Python 套接字制作程序

如何用 Python 制作一个简单的端口扫描程序？

这个小端口扫描程序将尝试连接到您为特定主机定义的每个端口。我们必须做的第一件事是导入套接字库和我们需要的其他库。

打开一个文本编辑器，复制并粘贴下面的代码。

将文件另存为:“portscanner.py”并退出编辑器

```py
#!/usr/bin/env python
import socket
import subprocess
import sys
from datetime import datetime

# Clear the screen
subprocess.call('clear', shell=True)

# Ask for input
remoteServer    = raw_input("Enter a remote host to scan: ")
remoteServerIP  = socket.gethostbyname(remoteServer)

# Print a nice banner with information on which host we are about to scan
print "-" * 60
print "Please wait, scanning remote host", remoteServerIP
print "-" * 60

# Check what time the scan started
t1 = datetime.now()

# Using the range function to specify ports (here it will scans all ports between 1 and 1024)

# We also put in some error handling for catching errors

try:
    for port in range(1,1025):  
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((remoteServerIP, port))
        if result == 0:
            print "Port {}: 	 Open".format(port)
        sock.close()

except KeyboardInterrupt:
    print "You pressed Ctrl+C"
    sys.exit()

except socket.gaierror:
    print 'Hostname could not be resolved. Exiting'
    sys.exit()

except socket.error:
    print "Couldn't connect to server"
    sys.exit()

# Checking the time again
t2 = datetime.now()

# Calculates the difference of time, to see how long it took to run the script
total =  t2 - t1

# Printing the information to screen
print 'Scanning Completed in: ', total 
```

##### 抽样输出

让我们运行程序，看看输出会是什么样子

```py
$ python portscanner.py

Enter a remote host to scan: www.your_host_example.com
------------------------------------------------------------
Please wait, scanning remote host xxxx.xxxx.xxxx.xxxx
------------------------------------------------------------

Port 21:   Open
Port 22:    Open
Port 23:    Open
Port 80:    Open
Port 110:   Open
Port 111:   Open
Port 143:   Open
Port 443:   Open
Port 465:   Open
Port 587:   Open
Port 993:   Open
Port 995:   Open

Scanning Completed in:  0:06:34.705170 
```

### 放弃

该程序旨在让个人测试他们自己的设备的弱安全性，如果它被用于任何其他用途，作者将不承担任何责任