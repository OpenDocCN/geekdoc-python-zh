# 如何使用 Pexpect 模块

> 原文：<https://www.pythonforbeginners.com/systems-programming/how-to-use-the-pexpect-module-in-python>

本文基于来自 http://www.noah.org/wiki/pexpect 和 http://pypi.python.org/pypi/pexpect/的[文档](http://www.noah.org/wiki/pexpect "noah.org_pexpect")

我开始使用 Pexepect 的原因是因为我在寻找一个可以满足我的一些自动化需求的模块(主要是 ssh 和 ftp)。

你可以使用其他模块比如[子流程](https://docs.python.org/2/library/subprocess.html "subprocess_python")，但是我发现这个模块更容易使用。

注意，这篇文章不是为 Python 初学者写的，但是学习新事物总是很有趣的

## 什么是 Pexpect？

Pexpect 是一个纯 Python 模块，使 Python 成为控制
和自动化其他程序的更好工具。

Pexpect 基本上是一个模式匹配系统。它运行程序并观察
输出。

当输出与给定的模式匹配时，Pexpect 可以做出响应，就像人类在
输入响应一样。

## Pexpect 可以用来做什么？

Pexpect 可以用于自动化、测试和屏幕抓取。

Pexpect 可用于自动化交互式控制台应用程序，如
ssh、ftp、passwd、telnet 等。

它还可以通过“lynx”、“w3m”或其他基于文本的网络浏览器来控制网络应用程序。

## 安装 Pexpect

Pexpect 的最新版本可以在[这里](https://sourceforge.net/projects/pexpect/files/pexpect/ "pexpect_dl")找到

```py
wget http://pexpect.sourceforge.net/pexpect-2.3.tar.gz
tar xzf pexpect-2.3.tar.gz
cd pexpect-2.3
sudo python ./setup.py install

# If your systems support yum or apt-get, you might be able to use the
# commands below to install the pexpect package. 

sudo yum install pexpect.noarch

# or

sudo apt-get install python-pexpect 
```

## 预期方法

Pexpect 中有两个重要的方法:expect()和 send()(或者 sendline()
，它类似于带有换行符的 send()。

#### expect()方法

等待子应用程序返回给定的强。

您指定的字符串是一个[正则表达式](https://www.pythonforbeginners.com/regex/regular-expressions-in-python)，因此您可以匹配复杂的
模式。

记住，任何时候你试图匹配一个需要前瞻的模式，你总是会得到一个最小匹配。

下面将总是返回一个字符:
child.expect('。+')

正确指定您期望返回的文本，您可以添加'.* '到文本的开头
或结尾，以确保捕捉到
个意外字符

**这个例子将匹配成功，但将始终不返回任何字符:**
child.expect('。*')

一般任何星号*表达式都会尽量少匹配。

给予 expect()的模式也可以是一列正则表达式，
这允许您匹配多个可选响应。
(例如，如果您从服务器获得各种响应)

#### send()方法

将字符串写入子应用程序。

从孩子的角度来看，这看起来就像有人从终端输入了文本。

#### 之前和之后的属性

在每次调用 expect()之后，before 和 After 属性将被设置为由子应用程序打印的
文本。

**before 属性**将包含预期字符串模式之前的所有文本。

您可以使用 child.before 来打印连接另一端的输出

**之后的字符串**将包含与预期模式匹配的文本。

**匹配属性**被设置为 re MatchObject。

## 从远程 FTP 服务器连接并下载文件

这将连接到 openbsd ftp 站点并下载递归目录
清单。

您可以在任何应用程序中使用这种技术。

如果您正在编写自动化测试工具，这尤其方便。

同样，这个例子是从[复制到这里](http://www.noah.org/wiki/Pexpect "noah.org_pexpect")

```py
import pexpect
child = pexpect.spawn ('ftp ftp.openbsd.org')
child.expect ('Name .*: ')
child.sendline ('anonymous')
child.expect ('Password:')
child.sendline ('[[email protected]](/cdn-cgi/l/email-protection)')
child.expect ('ftp> ')
child.sendline ('cd pub')
child.expect('ftp> ')
child.sendline ('get ls-lR.gz')
child.expect('ftp> ')
child.sendline ('bye') 
```

在第二个例子中，我们可以看到如何从 Pexpect 取回控制权

## 连接到远程 FTP 服务器并获得控制权

这个例子使用 ftp 登录到 OpenBSD 站点(如上所述)，
列出一个目录中的文件，然后将 ftp 会话的交互控制
传递给人类用户。

```py
import pexpect
child = pexpect.spawn ('ftp ftp.openbsd.org')
child.expect ('Name .*: ')
child.sendline ('anonymous')
child.expect ('Password:')
child.sendline ('[[email protected]](/cdn-cgi/l/email-protection)')
child.expect ('ftp> ')
child.sendline ('ls /pub/OpenBSD/')
child.expect ('ftp> ')
print child.before    # Print the result of the ls command.
child.interact()       # Give control of the child to the user. 
```

## EOF、超时和行尾

有特殊的模式来匹配文件结束或超时条件。

我不会在这篇文章中写它，而是参考官方的[文档](http://www.noah.org/wiki/pexpect "noah_pexpect2")
，因为知道它是如何工作的是很好的。