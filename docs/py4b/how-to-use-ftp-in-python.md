# 如何在 Python 中使用 FTP

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/how-to-use-ftp-in-python>

## 概观

本文将展示如何借助
ftplib 模块在 Python 中使用 FTP。

## Ftplib

Python 中的 ftplib 模块允许您编写执行各种自动化 FTP 作业的 Python 程序。您可以轻松地连接到 FTP 服务器来检索
文件并在本地处理它们。

要在 Python 中使用 ftplib 模块，首先必须将其导入到脚本中。

## 打开连接

要“打开”到 FTP 服务器的连接，您必须创建对象。

一旦建立了连接(打开了)，就可以使用 ftplib
模块中的方法。

有两种类型的方法:一种用于处理文本文件，另一种用于二进制文件。

您可以轻松浏览目录结构，管理和下载文件。

## 我如何使用它？

这个程序将首先连接到一个 ftp 服务器(ftp.cwi.nl ),然后使用 list()方法列出 FTP 服务器根目录中的
文件和目录。

```py
from ftplib import FTP

ftp = FTP('ftp.cwi.nl')   # connect to host, default port

ftp.login()               # user anonymous, passwd [[email protected]](/cdn-cgi/l/email-protection)

ftp.retrlines('LIST')     # list directory contents 
```

我们的第二个程序打开一个到“ftp.sunet.se”的连接，用户名为“anonymous”
，电子邮件地址为“[【受电子邮件保护】](/cdn-cgi/l/email-protection)

然后使用 dir()
方法列出 FTP 服务器上的文件和目录。

输出保存到“文件”变量中。

然后，我使用打印来查看屏幕上的文件。

如果我想更改目录，我只需使用 ftp.cwd(path)即可。

要关闭 FTP 连接，请使用 quit()方法。

```py
import ftplib

ftp = ftplib.FTP('ftp.sunet.se', 'anonymous', '[[email protected]](/cdn-cgi/l/email-protection)')

print "File List: "

files = ftp.dir()

print files

ftp.cwd("/pub/unix") #changing to /pub/unix 
```

## 常见的 FTP 方法

#### FTP.connect(主机[，端口[，超时]])

连接到给定的主机和端口。

根据 FTP 协议规范的规定，默认端口号为 21。

很少需要指定不同的端口号。

对于每个实例，该函数只应调用一次

如果在创建实例时给定了主机，则根本不应该调用它。

所有其他方法只能在连接
完成后使用。

可选的超时参数以秒为单位指定连接
尝试的超时。

如果没有超时，将使用全局默认超时设置。

#### FTP.getwelcome()

返回服务器在回复初始连接时发送的欢迎消息。

此消息有时包含可能与用户相关的免责声明或帮助信息

#### FTP . log in([用户[，密码[，帐户]])

以给定用户的身份登录。

passwd 和 acct 参数是可选的，默认为空字符串。

如果未指定用户，则默认为“匿名”。

如果用户为“匿名”，默认密码为“[【受电子邮件保护】](/cdn-cgi/l/email-protection)”。

在连接
建立之后，对于每个实例，这个函数应该只调用一次。

如果在创建实例
时给定了主机和用户，则根本不应该调用它。

大多数 FTP 命令只有在客户端登录后才被允许。

acct 参数提供“会计信息”；很少有系统实现这一点。

#### FTP.retrbinary(命令，回调[，maxblocksize[，rest]])

以二进制传输模式检索文件。

命令应该是适当的 RETR 命令:“RETR 文件名”。

对于接收到的每个数据块调用回调函数，用一个单独的
字符串参数给出数据块。

可选的 maxblocksize 参数指定在为进行实际传输而创建的低级套接字对象上读取的最大块大小。

选择一个合理的默认值。rest 的含义与 transfercmd()
方法中的含义相同。

#### FTP.retrlines(命令[，回调])

以 ASCII 传输模式检索文件或目录列表。

命令应该是一个适当的 RETR 命令或命令，如名单，NLST 或
MLSD。

LIST 检索文件列表和关于这些文件的信息。

NLST 检索文件名列表。

在一些服务器上，MLSD 检索机器可读的文件列表和关于这些文件的信息。

回调函数是为每一行调用的，每一行都有一个包含
的字符串参数，即去掉了尾部 CRLF 的那一行。

默认回调将该行打印到 sys.stdout。

#### FTP.dir(argument[, …])

生成 LIST 命令返回的目录列表，并将其打印到
标准输出。

可选参数是要列出的目录(默认为当前服务器的
目录)。

可以使用多个参数将非标准选项传递给 LIST 命令。

如果最后一个参数是一个函数，就像对于
retrlines()一样作为回调函数使用；默认打印到 sys.stdout。

此方法返回 None。

#### FTP.delete(文件名)

从服务器上删除名为 filename 的文件。

如果成功，返回响应的文本，否则在出现
权限错误时引发 error_perm，在出现其他错误时引发 error_reply。

#### FTP.cwd(路径名)

在服务器上设置当前目录。

#### FTP.mkd(路径名)

在服务器上创建一个新目录。

#### FTP.pwd()

返回服务器上当前目录的路径名。

#### FTP.quit()

向服务器发送退出命令并关闭连接。

这是关闭连接的“礼貌”方式，但如果服务器对 QUIT 命令做出错误响应，可能会引发异常。

这意味着对 close()方法的调用，这使得 FTP 实例对后续调用无用。

#### FTP.close()

单方面关闭连接。

这不应应用于已经关闭的连接，例如在
成功调用 quit()之后。

在这个调用之后，FTP 实例不应该再被使用。

调用 close()或 quit()后，不能通过发出
另一个 login()方法来重新打开连接。

更多信息，请参见官方 [Python 文档](https://docs.python.org/2/library/ftplib.html "pythondocs")