# Python 设置

> 原文：<https://www.pythonforbeginners.com/basics/python-setup>

Python 是免费和开源的，可用于 python.org 的所有操作系统。如果尚未安装，本文将帮助您安装 python。

Python 已经安装在许多操作系统上。

在这篇文章中，我使用了一些来自 Google 教育资料的例子，这些资料是在这里找到的[，作为帮助你设置 python 的来源之一。](https://developers.google.com/edu/python/set-up "google_setup_python")

## 我的电脑上安装 Python 了吗？

默认情况下，除 Windows 之外的大多数操作系统都已经安装了 Python

要检查 Python 是否已安装，请打开命令行(通常通过运行
**“终端”**程序)并键入**“Python-V”**

如果您看到一些版本信息，那么您已经安装了 Python。

然而，如果你得到**“bash:python:command not found”**，有可能
你的电脑上没有安装 Python。

请看看[这篇](http://www.diveintopython.net/installing_python/index.html#install.choosing "divintopython")在 diveintopython.net
发表的精彩帖子，他们描述了如何在各种操作系统上安装 Python。

## Python 解释器

当您以交互模式启动 Python 时，它会提示输入下一个命令，
通常是三个大于号(**>>>**)。

解释器在打印第一个提示之前打印一条欢迎消息，说明其版本号和一个
版权声明，例如:

要交互式运行 Python 解释器，只需在终端中键入**“Python”**:

```py
??$ python
Python 2.7.2 (default, Jun 20 2012, 16:23:33) 
[GCC 4.2.1 Compatible Apple Clang 4.0 (tags/Apple/clang-418.0.60)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 

>>> 1 + 1
2
>>> you can type expressions here .. use ctrl-d to exit 
```

## 我如何运行我的 Python 程序？

运行 python 程序最简单的方法是输入 **"python helloworld.py"**

如果在一个. py (python 扩展名)文件上设置了“执行位”,则可以通过
名称运行，而不必先键入**“python”**。

使用“chmod”命令设置执行位，如下所示:
**$ chmod +x hello.py.**

现在你可以把你的程序命名为**。/hello.py**

## 如何安装 python 窗口？

在 Windows 上做一个基本的 Python 安装很容易，只需进入[python.org](https://python.org "python.org")下载页面，选择一个版本，比如 2.7。

运行 Python 安装程序，采用所有默认值。

这将在根目录中安装 Python 并设置一些文件关联。

安装 Python 后，打开命令提示符(附件>命令提示符，或
在运行对话框中键入“cmd ”)

您应该可以通过键入“**python”**
然后键入**“hello world . py”**来运行 **helloworld.py** python 程序

要交互运行 python 解释器，从开始
菜单中选择**运行**命令，并输入 **"python"** ，这将在它自己的窗口中交互启动 Python。

在 Windows 上，使用 **Ctrl-Z 退出**(在所有其他操作系统上，使用 **Ctrl-D 退出**)

## 文本编辑器

在源文件中编写 Python 程序之前，我们需要一个编辑器来编写
源文件。

对于 Python 编程，有大量编辑器可供使用。

选择一个适合你的平台的。

你选择哪个编辑器取决于你对计算机有多少经验，
你需要它做什么，你需要在哪个平台上做。

至少，你需要一个对代码
和缩进(Notepad ++，Smultron，TextEdit，Gedit，Kate，Vi 等)有一点了解的文本编辑器..)

## 我如何编写 Python 程序

Python 程序只是一个可以直接编辑的文本文件。

在命令行(或终端)中，你可以运行任何你想运行的 python 程序，就像上面我们对**“python hello world . py”**所做的那样

在命令行提示符下，只需按向上箭头键即可调出之前键入的
命令，因此无需重新键入即可轻松运行之前的命令。

要试用您的编辑器，请编辑 **helloworld.py** 程序。

将代码中的**【你好】**改为**【您好】**。

保存您的编辑并运行程序以查看其新输出。

尝试添加一个**“打印‘耶！’”**就在现有的印刷品下面，并带有同样的
缩进。

尝试运行该程序，以查看您的编辑工作是否正常。

### 相关职位

[Python 教程](https://www.pythonforbeginners.com/python-tutorial/)

[Python 环境设置](https://www.pythonforbeginners.com/learn-python/python-environment-setup)