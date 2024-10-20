# Python 101:使用 ftplib 下载文件

> 原文：<https://www.blog.pythonlibrary.org/2012/07/19/python-101-downloading-a-file-with-ftplib/>

使用 Python 从互联网下载文件有很多不同的方法。一种流行的方法是连接到 FTP 服务器，并通过这种方式下载文件。这就是我们将在本文中探讨的内容。您所需要的只是 Python 的标准安装。它包括一个名为 ftplib 的库，其中包含了我们完成这项任务所需的所有信息。

## 下载吧！

下载文件其实很容易。下面是一个简单的例子来说明如何做到这一点:

```py

# ftp-ex.py

import os
from ftplib import FTP

ftp = FTP("www.myWebsite.com", "USERNAME", "PASSWORD")
ftp.login()
ftp.retrlines("LIST")

ftp.cwd("folderOne")
ftp.cwd("subFolder") # or ftp.cwd("folderOne/subFolder")

listing = []
ftp.retrlines("LIST", listing.append)
words = listing[0].split(None, 8)
filename = words[-1].lstrip()

# download the file
local_filename = os.path.join(r"c:\myfolder", filename)
lf = open(local_filename, "wb")
ftp.retrbinary("RETR " + filename, lf.write, 8*1024)
lf.close()

```

让我们把它分解一下。首先，我们需要登录到 FTP 服务器，所以您将传递 URL 以及您的凭证，或者如果它是那些匿名 FTP 服务器之一，您可以跳过它。retrlines("LIST") 命令将给出一个目录列表。 **cwd** 命令代表“更改工作目录”，所以如果当前目录没有你要找的，你需要使用 cwd 来更改到有你要找的目录。下一节将展示如何以一种相当愚蠢的方式获取文件名。在大多数情况下，你也可以使用 **os.path.basename** 得到同样的结果。最后一节展示了如何实际下载文件。注意，我们必须打开带有“wb”(写二进制)标志的文件处理程序，这样我们才能正确下载文件。“8*1024”位是下载的块大小，尽管 Python 很聪明地选择了一个合理的默认值。

 ***注意:**本文基于 [ftplib 模块](http://docs.python.org/library/ftplib.html)的 Python 文档，以及您的默认 Python 安装文件夹中的以下脚本:Tools/scripts/ftpmirror.py.*

### 进一步阅读

*   Python 101: [如何下载文件](https://www.blog.pythonlibrary.org/2012/06/07/python-101-how-to-download-a-file/)
*   ftplib [正式文档](http://docs.python.org/library/ftplib.html)