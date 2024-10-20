# python 101:ftplib 简介

> 原文：<https://www.blog.pythonlibrary.org/2016/06/23/python-101-an-intro-to-ftplib/>

许多公司和组织使用文件传输协议(FTP)来共享数据。Python 在其标准库中提供了一个名为 **ftplib** 的文件传输协议模块，实现 FTP 协议的客户端。您可以通过阅读互联网上的 RFC 959 文档来了解文件传输协议的所有内容。然而，完整的规范超出了本文的范围。相反，我们将关注以下主题:

*   连接到 FTP 服务器
*   浏览它的结构
*   从 FTP 服务器下载文件
*   将文件上传到 FTP 服务器

我们开始吧！

* * *

### 连接到 FTP 服务器

我们需要做的第一件事是找到一个要连接的 FTP 服务器。有很多免费的可以用。例如，大多数 Linux 发行版都有可以公开访问的 FTP 镜像。如果你去 Fedora 的网站([https://admin.fedoraproject.org/mirrormanager/](https://admin.fedoraproject.org/mirrormanager/))你会发现一长串你可以使用的镜子。但是它们不仅仅是 FTP，所以请确保您选择了正确的协议，否则您将会收到连接错误。

对于这个例子，我们将使用**ftp.cse.buffalo.edu**。官方的 Python 文档使用 ftp.debian.org 的 T2，所以也可以随意尝试一下。现在让我们尝试连接到服务器。在您的终端中打开 Python 解释器或使用 IDLE 跟随:

```py

>>> from ftplib import FTP
>>> ftp = FTP('ftp.cse.buffalo.edu')
>>> ftp.login()
'230 Guest login ok, access restrictions apply.'

```

让我们把它分解一下。这里我们从 ftplib 导入**FTP**类。然后，我们创建一个类的实例，将它传递给我们想要连接的主机。因为我们没有传递用户名或密码，Python 假设我们想要匿名登录。如果您碰巧需要使用非标准端口连接到 FTP 服务器，那么您可以使用 **connect** 方法。方法如下:

```py

>>> from ftplib import FTP
>>> ftp = FTP()
>>> HOST = 'ftp.cse.buffalo.edu'
>>> PORT = 12345
>>> ftp.connect(HOST, PORT)

```

这段代码将会失败，因为这个例子中的 FTP 服务器没有为我们打开端口 12345。然而，这个想法是传达如何连接到一个不同于默认的端口。

如果你正在连接的 FTP 服务器需要 TLS 安全性，那么你将需要导入 **FTP_TLS** 类，而不是 **FTP** 类。 **FTP_TLS** 类支持密钥文件和证书文件。如果您想要保护您的连接，那么您将需要调用 **prot_p** 来这样做。

* * *

### 使用 ftplib 导航目录

让我们来学习如何查看 FTP 服务器上的内容并更改目录！下面是一些代码，演示了这样做的正常方法:

```py

>>> from ftplib import FTP
>>> ftp = FTP()
>>> ftp.login()
>>> ftp.retrlines('LIST')   
total 28
drwxrwxrwx   2 0       0     4096 Sep  6  2015 .snapshot
drwxr-xr-x   2 202019  5564  4096 Sep  6  2015 CSE421
drwxr-xr-x   2 0       0     4096 Jul 23  2008 bin
drwxr-xr-x   2 0       0     4096 Mar 15  2007 etc
drwxr-xr-x   6 89987   546   4096 Sep  6  2015 mirror
drwxrwxr-x   7 6980    546   4096 Jul  3  2014 pub
drwxr-xr-x  26 0       11    4096 Apr 29 20:31 users
'226 Transfer complete.'
>>> ftp.cwd('mirror')
'250 CWD command successful.'
>>> ftp.retrlines('LIST')   
total 16
drwxr-xr-x  3 89987  546  4096 Sep  6  2015 BSD
drwxr-xr-x  5 89987  546  4096 Sep  6  2015 Linux
drwxr-xr-x  4 89987  546  4096 Sep  6  2015 Network
drwxr-xr-x  4 89987  546  4096 Sep  6  2015 X11
'226 Transfer complete.'

```

我们在这里登录，然后向 FTP 服务器发送 LIST 命令。这是通过调用我们的 ftp 对象的 **retrlines** 方法来完成的。 **retrlines** 方法打印出我们调用的命令的结果。在这个例子中，我们调用 LIST 来检索文件和/或文件夹的列表以及它们各自的信息并打印出来。然后我们使用 **cwd** 命令将我们的工作目录更改到一个不同的文件夹，然后重新运行 LIST 命令来查看其中的内容。您也可以使用 ftp 对象的 **dir** 函数来获取当前文件夹的列表。

* * *

### 通过 FTP 下载文件

仅仅查看 FTP 服务器上的内容并没有那么有用。您几乎总是想从服务器下载文件。让我们看看如何下载单个文件:

```py

>>> from ftplib import FTP
>>> ftp = FTP('ftp.debian.org')
>>> ftp.login()
'230 Login successful.'
>>> ftp.cwd('debian')  
'250 Directory successfully changed.'
>>> out = '/home/mike/Desktop/README'
>>> with open(out, 'wb') as f:
...     ftp.retrbinary('RETR ' + 'README.html', f.write)

```

在这个例子中，我们登录到 Debian Linux FTP 并切换到 Debian 文件夹。然后，我们创建想要保存的文件的名称，并以写二进制模式打开它。最后，我们使用 ftp 对象的 **retrbinary** 调用 RETR 来检索文件并将其写入本地磁盘。如果你想下载所有的文件，那么我们需要一个文件列表。

```py

import ftplib
import os

ftp = ftplib.FTP('ftp.debian.org')
ftp.login()
ftp.cwd('debian')
filenames = ftp.nlst()

for filename in filenames:
    host_file = os.path.join(
        '/home/mike/Desktop/ftp_test', filename)
    try:
        with open(host_file, 'wb') as local_file:
            ftp.retrbinary('RETR ' + filename, local_file.write)
    except ftplib.error_perm:
        pass

ftp.quit()

```

这个例子和上一个很相似。不过，您需要修改它以匹配您自己的首选下载位置。代码的第一部分非常相似，但是你会注意到我们调用了 **nlst** ，它给出了文件名和目录的列表。你可以给它一个目录列表，或者直接调用它，它会假设你想要一个当前目录的列表。注意，nlst 命令没有告诉我们如何从结果中区分文件和目录。但是对于这个例子，我们根本不关心。这更像是一个暴力脚本。所以它将遍历返回的列表并尝试下载它们。如果“文件”实际上是一个目录，那么我们将在本地磁盘上创建一个与 FTP 服务器上的目录同名的空文件。

有一个 MLSD 命令可以通过 **mlsd** 方法调用，但是并不是所有的 FTP 服务器都支持这个命令。如果是这样，那么你也许能够区分这两者。

* * *

### 将文件上传到 FTP 服务器

FTP 服务器的另一个主要任务是向它上传文件。Python 也可以处理这个问题。实际上有两种方法可以用来上传文件:

*   storlines -用于上传文本文件(TXT，HTML，RST)
*   storbinary -用于上传二进制文件(PDF，XLS 等)

让我们来看一个如何实现这一点的示例:

```py

import ftplib

def ftp_upload(ftp_obj, path, ftype='TXT'):
    """
    A function for uploading files to an FTP server
    @param ftp_obj: The file transfer protocol object
    @param path: The path to the file to upload
    """
    if ftype == 'TXT':
        with open(path) as fobj:
            ftp.storlines('STOR ' + path, fobj)
    else:
        with open(path, 'rb') as fobj:
            ftp.storbinary('STOR ' + path, fobj, 1024)

if __name__ == '__main__':
    ftp = ftplib.FTP('host, 'username', 'password')
    ftp.login()

    path = '/path/to/something.txt'
    ftp_upload(ftp, path)

    pdf_path = '/path/to/something.pdf'
    ftp_upload(ftp, pdf_path, ftype='PDF')

    ftp.quit()

```

在这个例子中，我们创建了一个上传文件的函数。它接受一个 ftp 对象、我们想要上传的文件的路径和文件的类型。然后，我们对文件类型进行快速检查，以确定我们是否应该使用 **storlines** 或 **storbinary** 进行上传。最后，在底部的条件语句中，我们连接到 FTP 服务器，登录并上传一个文本文件和一个 PDF 文件。一个简单的增强功能是在我们登录后切换到一个特定的目录，因为我们可能不希望只是将文件上传到根目录。

* * *

### 包扎

至此，您应该对 Python 的 ftplib 有了足够的了解，可以开始使用了。它还有很多其他方法，值得在 Python 的文档中查看。但是您现在已经知道了列出目录、浏览文件夹结构以及下载和上传文件的基本知识。

* * *

### 相关阅读

*   Python 文档: [ftplib](https://docs.python.org/3/library/ftplib.htm)
*   eff bot-[ftplib 模块](http://effbot.org/librarybook/ftplib.htm)
*   Python ftplib [教程](https://pythonprogramming.net/ftp-transfers-python-ftplib/)