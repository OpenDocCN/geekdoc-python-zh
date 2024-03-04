# Python 中的子流程和 Shell 命令

> 原文：<https://www.pythonforbeginners.com/os/subprocess-for-system-administrators>

## 子流程概述

很长一段时间以来，我一直使用 os.system()来处理 Python 中的系统
管理任务。

主要原因是，我认为这是运行 Linux 命令最简单的方式。

在官方的 python 文档中，我们可以看到应该使用子进程
来访问系统命令。

子流程模块允许我们生成流程，连接到它们的
输入/输出/错误管道，并获得它们的返回代码。

子进程旨在替换其他几个较旧的模块和函数，如:os.system、os.spawn*、os.popen*、popen2。*命令。

让我们开始研究子流程的不同功能。

## subprocess.call()

运行“args”描述的命令。

我们可以运行命令行，将参数作为字符串列表
(示例 1)传递，或者将 shell 参数设置为 True 值(示例 2)

注意，shell 参数的默认值是 False。

让我们看两个例子，在这两个例子中，我们使用
subprocess.call()显示了磁盘使用情况的摘要

```py
subprocess.call(['df', '-h']) 
```

这一次我们将 shell 参数设置为 True

```py
subprocess.call('du -hs $HOME', shell=True) 
```

注意，官方 Python 文档对使用
shell=True 参数提出了警告。

“使用 shell=True 调用系统 shell，如果将
与不受信任的输入结合起来，可能会有安全隐患”[ [来源](https://docs.python.org/2/library/subprocess.html "subprocess_python")

现在，让我们继续，看看输入/输出。

## 输入和输出

使用 subprocess，您可以取消输出，这在您想要运行系统调用但对标准输出不感兴趣时非常方便。

它还为您提供了一种方法，在以标准方式管理输入/输出的同时，将 shell 命令干净地集成到您的脚本
中。

## 返回代码

您可以使用 subprocess.call 返回代码来确定命令是否成功。

每个进程都会返回一个退出代码，你可以根据这个代码用你的脚本
做一些事情。

如果返回代码不是零，则意味着发生了错误。

如果想用 Python 做系统管理，推荐阅读
[Python for Unix 和 Linux 系统管理](https://amzn.to/3gYVmit "pythonforadmins")

## stdin, stdout and stderr

我在 subprocess 中遇到的最棘手的部分之一是如何使用管道
和管道命令。

管道表示应该创建一个到子管道的新管道。

默认设置是“无”，这意味着不会发生重定向。

标准错误(或 stderr)可以是 stdout，这表明来自子进程的 stderr
数据应该被捕获到与 STDOUT 的
相同的文件句柄中。

## 子流程。波本()

子流程模块中的底层流程创建和管理由 Popen 类
处理。subprocess.popen 正在替换 os.popen。

让我们从一些真实的例子开始。

子流程。Popen 接受一个参数列表

```py
import subprocess

p = subprocess.Popen(["echo", "hello world"], stdout=subprocess.PIPE)

print p.communicate()

>>>('hello world', None) 
```

注意，尽管您可以使用“shell=True ”,但这不是推荐的方式。

如果您知道您将只使用特定的子流程函数，比如 Popen 和 PIPE，那么只导入它们就足够了。

```py
from subprocess import Popen, PIPE

p1 = Popen(["dmesg"], stdout=PIPE)

print p1.communicate() 
```

## Popen.communicate()

communicate()方法返回一个元组(stdoutdata，stderrdata)。

Popen.communicate()与 process: Send data to stdin 交互。

从 stdout 和 stderr 读取数据，直到到达文件结尾。

等待进程终止。

可选的输入参数应该是一个发送到
子进程的字符串，如果没有数据发送到子进程，则为 None。

基本上，当您使用 communicate()时，这意味着您想要
执行命令

## 使用子进程的 Ping 程序

在下面的“更多阅读”部分，您可以找到链接来阅读更多关于子流程模块的
，以及示例。

让我们编写自己的 ping 程序，首先向用户请求输入，
，然后对该主机执行 ping 请求。

```py
# Import the module
import subprocess

# Ask the user for input
host = raw_input("Enter a host to ping: ")	

# Set up the echo command and direct the output to a pipe
p1 = subprocess.Popen(['ping', '-c 2', host], stdout=subprocess.PIPE)

# Run the command
output = p1.communicate()[0]

print output 
```

让我们再举一个例子。这次我们使用 host 命令。

```py
target = raw_input("Enter an IP or Host to ping:
")

host = subprocess.Popen(['host', target], stdout = subprocess.PIPE).communicate()[0]

print host 
```

我建议您阅读下面的链接，以获得更多关于 Python 中的
子流程模块的知识。

如果您有任何问题或意见，请使用下面的评论栏。

##### 更多阅读

[如何在 Python 中 sh](https://www.pythonforbeginners.com/modules-in-python/how-to-use-sh-in-python)
[http://docs.python.org/2/library/subprocess.html](https://docs.python.org/2/library/subprocess.html "subprocess_docs")
[曾经有用而整洁的子流程模块](http://sharats.me/the-ever-useful-and-neat-subprocess-module.html "useful_subprocess")
[http://pymotw.com/2/subprocess/](http://pymotw.com/2/subprocess/ "subprocess.pymotw")
[http://www . bogoto bogo . com/Python/Python _ subprocess _ module . PHP](http://www.bogotobogo.com/python/python_subprocess_module.php "bogotobogo.com")