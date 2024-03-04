# 在 Python 中使用平台模块

> 原文：<https://www.pythonforbeginners.com/basics/how-to-use-the-platform-module-in-python>

## 它是用来做什么的？

Python 中的平台模块用于访问底层平台的数据，例如硬件、操作系统和解释器版本信息。

平台模块包括查看平台硬件、操作
系统和程序运行的解释器版本信息的工具。

## 我如何使用它？

你从在你的程序中导入平台模块开始，就像这样:
**导入平台** 
然后你指定你想要查找的内容(更多内容在下面)。

举个例子，如果你想知道你用的是哪个 python 版本，
只需在平台上添加 python_version()，就像这样:
**打印 platform.python_version()**

这将返回字符串形式的 Python 版本。

在我的电脑上看起来是这样的:
**2.7.3**

## 平台功能

让我们来看看我们可以使用的不同平台功能

**platform . architecture()** 返回位架构信息

**platform.machine()** 返回机器类型，如' i386 '。

**platform.node()** 返回计算机的网络名称(可能不是全限定！)

**platform.platform()**
返回一个标识底层平台的字符串，其中包含尽可能多的有用信息
。

**platform.processor()** 返回(真实的)处理器名，如' amdk6 '。

**platform . python _ build()**
返回一个元组(buildno，builddate ),以字符串形式表示 Python 编译号和
日期。

**platform . python _ compiler()**
返回一个字符串，标识用于编译 Python 的编译器。

**platform . python _ version()**
以字符串' major.minor.patchlevel '的形式返回 Python 版本

**platform . python _ implementation()**
返回标识 Python 实现的字符串。
可能的返回值有:' CPython '，' IronPython '，' Jython '，' PyPy '。

**platform.release()**
返回系统的版本，如“2.2.0”或“NT”

**platform.system()**
返回系统/操作系统名称，例如“Linux”、“Windows”或“Java”。

**platform.version()**
返回系统的发布版本，例如“德加的#3”

**platform.uname()**
返回一个字符串元组(系统、节点、版本、版本、机器、处理器)
标识底层平台。

## 操作系统和硬件信息

让我们展示一些带有输出的例子，看看这是如何工作的。

(你可以在这里找到更多例子)

```py
import platform

print 'uname:', platform.uname()

print
print 'system   :', platform.system()
print 'node     :', platform.node()
print 'release  :', platform.release()
print 'version  :', platform.version()
print 'machine  :', platform.machine()
print 'processor:', platform.processor() 
```

#### 输出

uname: ('Linux '，' Mwork '，' 3.5.0-21-generic '，' 32-Ubuntu SMP Tue Dec 11 18:51:59
UTC 2012 '，' x86_64 '，' x86_64 ')

系统:Linux
节点:Mwork
版本:3 . 5 . 0-21-通用
版本:# 32-Ubuntu SMP Tue 11 Dec 11 18:51:59 UTC 2012
机器:x86_64
处理器:x86_64

##### 来源

请查看以下网址了解更多信息

[http://docs.python.org/2/library/platform.html](https://docs.python.org/2/library/platform.html "http://docs.python.org/2/library/platform.html")
http://www.doughellmann.com/PyMOTW/platform/