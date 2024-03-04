# 使用 Python 的 SSH 连接

> 原文：<https://www.pythonforbeginners.com/code-snippets-source-code/ssh-connection-with-python>

## 概观

上周，我写了一篇关于 Python 中的 pexpect 模块以及如何使用它来满足一些自动化需求的文章，比如 ssh 和 ftp。

我想继续这个话题，写一写它的 pxssh 类。有了 pxshmodule，就可以很容易地通过 ssh 访问其他服务器。本文基于这里找到的官方文档:[http://pexpect.sourceforge.net/pxssh.html](http://pexpect.sourceforge.net/pxssh.html "pxssh_docs")

## pxssh 是什么？

Pxssh 基于 pexpect。它的类扩展了 pexpect.spawn 来专门设置 SSH 连接。我经常使用 pxssh 在 python 中建立 ssh 连接。

## 模块文档

打开一个终端，键入以下命令以获得关于该模块的帮助

```py
import pxssh
help(pxssh)

Help on module pxssh:

NAME
   pxssh

FILE
   /usr/lib/python2.7/dist-packages/pxssh.py

DESCRIPTION
   This class extends pexpect.spawn to specialize setting up SSH connections.
   This adds methods for login, logout, and expecting the shell prompt.

   $Id: pxssh.py 513 2008-02-09 18:26:13Z noah $

CLASSES
   pexpect.ExceptionPexpect(exceptions.Exception)
       ExceptionPxssh
   pexpect.spawn(__builtin__.object)
       pxssh 
```

你也可以在这里看到帮助[http://pexpect.sourceforge.net/pxssh.html](http://pexpect.sourceforge.net/pxssh.html "pxssh_docs1")

## 方法和登录过程

Pxssh 添加了登录、注销和期望 shell 提示符的方法。在 SSH 登录过程中，它会处理各种棘手的情况。

例如，如果会话是您的第一次登录，那么 pxssh 会自动接受远程证书；或者，如果您设置了公钥认证，那么 pxssh 不会等待密码提示。

## pxssh 是如何工作的？

pxssh 使用 shell 提示符来同步远程主机的输出。为了使其更加健壮，它将 shell 提示符设置为比$或#更独特的东西。

这应该适用于大多数 Borne/Bash 或 Csh 风格的 shells。

## 例子

此示例在远程服务器上运行几个命令并打印结果。

首先，我们导入我们需要的模块。(pxssh 和 getpass)

我们导入了 getpass 模块，它会提示用户输入密码，而不会将用户输入的内容回显到控制台。

```py
 import pxssh
import getpass
try:                                                            
    s = pxssh.pxssh()
    hostname = raw_input('hostname: ')
    username = raw_input('username: ')
    password = getpass.getpass('password: ')
    s.login (hostname, username, password)
    s.sendline ('uptime')   # run a command
    s.prompt()             # match the prompt
    print s.before          # print everything before the prompt.
    s.sendline ('ls -l')
    s.prompt()
    print s.before
    s.sendline ('df')
    s.prompt()
    print s.before
    s.logout()
except pxssh.ExceptionPxssh, e:
    print "pxssh failed on login."
    print str(e) 
```

## 在远程 SSH 服务器上运行命令

让我们再举一个例子。要运行一个命令(“uptime”)并打印输出，您需要像这样做:

```py
import pxssh
s = pxssh.pxssh()
if not s.login ('localhost', 'myusername', 'mypassword'):
    print "SSH session failed on login."
    print str(s)
else:
    print "SSH session login successful"
    s.sendline ('uptime')
    s.prompt()         # match the prompt
    print s.before     # print everything before the prompt.
    s.logout()

#We can also execute multiple command like this:
s.sendline ('uptime;df -h') 
```

关于 pxssh 的更多信息，请参见官方[文档](http://pexpect.sourceforge.net/pxssh.html "pxssh_off_docs")