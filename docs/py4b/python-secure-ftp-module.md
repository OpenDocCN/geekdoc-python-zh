# Python 安全 FTP 模块

> 原文：<https://www.pythonforbeginners.com/modules-in-python/python-secure-ftp-module>

## 概观

在上一篇文章中，我们介绍了 Python 中的 ftplib 模块，你可以在这里阅读
更多关于[的内容。在这篇文章中，我们将介绍 pysftp 模块。](https://www.pythonforbeginners.com/code-snippets-source-code/how-to-use-ftp-in-python "ftplib")

SFTP(安全文件传输协议)用于通过互联网安全地交换文件
。

## 这是什么？

pysftp 是一个易于使用的 sftp 模块，它利用了 paramiko 和 pycrypto。

它提供了一个简单的 sftp 接口。

**一些特性是:**
自动优雅地处理 RSA 和 DSS 私钥文件

支持加密的私钥文件。

现在可以启用/禁用日志记录

## 我为什么要用它？

当您希望通过互联网安全地交换文件时。

## 我如何安装它？

pysftp 列在 PyPi 上，可以使用 pip 安装。

```py
# Search for pysftp
pip search pysftp

pysftp                    # - A friendly face on SFTP

#Install pysftp
pip install pysftp 
```

## 我如何使用它？

使用 pysftp 很容易，我们将展示一些如何使用它的例子

### 列出远程目录

要连接到我们的 FTP 服务器，我们首先必须导入 pysftp 模块并
指定(如果适用)服务器、用户名和密码凭证。

运行这个程序后，您应该会看到 FTP 服务器当前目录下的所有文件和目录。

```py
import pysftp

srv = pysftp.Connection(host="your_FTP_server", username="your_username",
password="your_password")

# Get the directory and file listing
data = srv.listdir()

# Closes the connection
srv.close()

# Prints out the directories and files, line by line
for i in data:
    print i 
```

## 连接参数

没有给出的参数是从环境中猜测出来的。

### **主持人**

远程机器的主机名。

### **用户名**

您在远程机器上的用户名。(无)

### **私钥**

您的私钥文件。(无)

### **密码**

您在远程机器上的密码。(无)

### **端口**

远程机器的 SSH 端口。(22)

### **私人钥匙通行证**

如果您的 private_key 已加密，则使用密码(无)

### **日志**

记录连接/握手详细信息(假)

### 下载/上传远程文件

和前面的例子一样，我们首先导入 pysftp 模块并指定
(如果适用)服务器、用户名和密码凭证。

我们还导入了 sys 模块，因为我们希望用户指定要下载/上传的文件。

```py
import pysftp
import sys

# Defines the name of the file for download / upload
remote_file = sys.argv[1]

srv = pysftp.Connection(host="your_FTP_server", username="your_username",
password="your_password")

# Download the file from the remote server
srv.get(remote_file)

# To upload the file, simple replace get with put. 
srv.put(remote_file)

# Closes the connection
srv.close() 
```

## 下一步是什么？

摆弄剧本，改变一些东西，看看会发生什么。

请尝试为其添加错误处理。如果没有传递参数会发生什么？

通过提示输入给程序增加一些交互。

##### 来源

[https://code.google.com/p/pysftp/](https://code.google.com/p/pysftp/ "pysftp")
http://en.wikipedia.org/wiki/SSH_File_Transfer_Protocol