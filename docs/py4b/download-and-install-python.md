# 下载并安装 Python

> 原文：<https://www.pythonforbeginners.com/basics/download-and-install-python>

## 概观

Python 是一种解释型语言，这意味着在程序运行时，代码被翻译成二进制代码。

这与编译语言(C++等)不同。)其中代码首先被
编译成二进制代码。

要运行 Python 代码，你需要一个 Python 解释器。Python 有不同的
版本，不是 Python 2 就是 Python 3。要了解差异并决定使用哪一个，请查看 python.org 的[这个](https://wiki.python.org/moin/Python2orPython3 "moin")维基页面。

## 安装 Python

Python 可以在大多数操作系统上使用(Linux、Unix、Mac OS X 和 Windows)

在你的电脑上安装它非常容易，在一些系统上已经有了。要查看它是否已经安装，打开一个终端并运行下面的命令
。

如果您看到来自 Python 解释器的响应，它会在初始显示中包含一个版本号
。

```py
>> python
Python 2.7.2 (default, Jun 20 2012, 16:23:33) 
```

如果你没有安装 Python，你可以看看这个链接，看看如何在你使用的平台上安装它。[http://www.diveintopython.net/installing_python/index.html](http://www.diveintopython.net/installing_python/index.html "DiveIntoPython")

## 我如何运行我的代码？

在 Python 中有两种运行程序的方法。

要么直接在 Python shell 中键入代码。这样做的时候
你会看到你输入的每个命令的结果。

这最适合非常短的程序或测试目的。

另一种运行代码的方式是在脚本中。

## Python Shell

当你在 Python Shell 中时，Python 解释器会为你翻译所有代码。

要离开帮助模式并返回到解释器，我们使用 quit 命令。

help 命令提供了一些关于 Python 的帮助

```py
>>> help
Type help() for interactive help, or help(object) for help about object.
>>> 
```

你也可以使用 Python Shell 来做数学(请参见我之前的[关于在 Python 中使用数学的文章](https://www.pythonforbeginners.com/basics/using-math-in-python "using_math"))

```py
>>> 2 + 4
6
>>> 5 * 56
280
>>> 5 - 45
-40
>>> 
```

要退出 Python Shell，请按 Ctrl+d。

## Python 脚本

要将程序作为脚本运行，请打开文本编辑器(vi、pico、nano 等)。)
并放入下面的代码:

```py
#!/usr/bin/python  
print "hello world" 
```

将文件另存为 hello.py 并退出编辑器。

```py
# To execute the program, type python and the file name in the shell. 
$python hello.py 
```

输出应该是:
hello world

Python 脚本可以像 shell 脚本一样直接执行，方法是在脚本的开头加上
shebang，并给文件一个可执行模式。

shebang 意味着当您想从 shell 中执行脚本时，脚本可以识别解释器类型。

```py
# The script can be given an executable mode, or permission, using the chmod command:
$ chmod +x hello.py 
```

现在，您可以直接通过脚本的名称来执行脚本。