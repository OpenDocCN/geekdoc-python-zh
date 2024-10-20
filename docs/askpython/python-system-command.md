# Python 系统命令:如何在 Python 中执行 Shell 命令？

> 原文：<https://www.askpython.com/python-modules/python-system-command>

## 介绍

今天在本教程中，我们将讨论如何使用 Python 系统命令来执行 shell 命令。

因此，让我们从 Python 系统命令的一些基础知识开始。

## 什么是 Python 系统命令？

我们可能需要集成一些特性来执行 Python 中的一些系统管理任务。这些包括查找文件、运行一些 shell 命令、进行一些高级文件处理等。为此，我们需要某种方式在系统和 python 解释器之间建立接口。

使用 Python 执行**命令行**可以通过使用`os module`中的一些系统方法轻松完成。

但是随着`subprocess`模块的引入(旨在取代一些旧的模块)，访问命令行变得更加容易使用。以及操纵输出和避免传统方法的一些限制。

## 在 Python 中执行 Shell 命令

现在我们已经了解了 Python 中的系统命令。让我们来看看我们如何实现同样的目标。

## 1.使用 os.system()方法

如前所述，使用`os`模块的一些方法可以很容易地在 Python 中执行 shell 命令。在这里，我们将使用广泛使用的`os.system()`方法。

该函数是使用**C**函数实现的，因此具有相同的限制。

该方法将系统命令作为字符串输入，并返回退出代码。

在下面的例子中，我们尝试使用 Python 中的命令行来检查我们的系统 **Python 版本**。

```py
import os

command = "python --version" #command to be executed

res = os.system(command)
#the method returns the exit status

print("Returned Value: ", res)

```

**输出:**

```py
Python 3.7.4
Returned Value:  0

```

这里，`res`存储返回值(退出代码= **0** 表示成功)。从输出中可以清楚地看到，命令执行成功，我们得到了预期的 Python 版本。

## 2.使用子流程模块

`subprocess`模块附带了各种有用的方法或函数来产生新的进程，连接到它们的输入/输出/错误管道，并获得它们的返回代码。

在本教程中，我们考虑的是`call()`和`check_output()`方法，因为它们是**易用**和**可靠**。但是要了解更多信息，你可以参考官方文档。

### 2.1.call()方法

现在进入`subprocess.call()`方法。

`call()`方法接受作为字符串列表传递的命令行参数，或者将 **shell** 参数设置为`True`。并返回给我们**退出代码**或**状态**。

在下面的代码片段中，我们尝试使用来自 **shell** 的 **PIP** 来安装 [pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 。

```py
import subprocess

command = "pip install pandas" #command to be executed

res = subprocess.call(command, shell = True)
#the method returns the exit code

print("Returned Value: ", res)

```

**输出:**

```py
Collecting pandas
  Downloading pandas-1.0.3-cp37-cp37m-win32.whl (7.5 MB)
Requirement already satisfied: pytz>=2017.2 in c:\users\sneha\appdata\local\programs\python\python37-32\lib\site-packages (from pandas) (2019.3)
Requirement already satisfied: numpy>=1.13.3 in c:\users\sneha\appdata\local\programs\python\python37-32\lib\site-packages (from pandas) (1.18.1)
Requirement already satisfied: python-dateutil>=2.6.1 in c:\users\sneha\appdata\local\programs\python\python37-32\lib\site-packages (from pandas) (2.8.1)
Requirement already satisfied: six>=1.5 in c:\users\sneha\appdata\local\programs\python\python37-32\lib\site-packages (from python-dateutil>=2.6.1->pandas) (1.14.0)
Installing collected packages: pandas
Successfully installed pandas-1.0.3
Returned Value:  0

```

正如我们所见，该命令成功执行，返回值为`zero`。

### 2.2.check output()方法

上面提到的方法成功地执行了传递的 shell 命令，但是没有给用户操作输出方式的自由。为此，**子流程的** `check_output()`方法必须参与进来。

该方法执行传递的**命令**，但是这次它没有返回退出状态，而是返回了一个`bytes`对象。

仔细看看下面的例子，我们试图再次安装`pymysql`模块(已经安装)。

```py
import subprocess

command = "pip install pymysql" #command to be executed

res = subprocess.check_output(command) #system command

print("Return type: ", type(res)) #type of the value returned

print("Decoded string: ", res.decode("utf-8")) #decoded result

```

**输出:**

```py
Return type:  <class 'bytes'>
Decoded string:  Requirement already satisfied: pymysql in c:\users\sneha\appdata\local\programs\python\python37-32\lib\site-packages (0.9.3)

```

这里类似于前面的情况，`res`通过`check_output()`方法保存返回的对象。我们可以看到`type(res)`确认了这个物体是`bytes`类型的。

之后，我们打印出**解码后的**字符串，并看到命令被成功执行。

## 结论

所以，今天我们学习了如何使用 Python 系统命令(os.system())和子进程模块来执行系统命令。我们在这里考虑了一些更多的与 python 相关的命令，但是值得注意的是方法并不局限于这些。

我们建议您自己使用上述方法尝试其他命令，以获得更好的理解。

任何进一步的问题，请在下面随意评论。

## 参考

*   [Python 子流程文档](https://docs.python.org/3/library/subprocess.html#module-subprocess)
*   [Python os 文档](https://docs.python.org/3/library/os.html?highlight=os%20system#os.system)，
*   Python 系统命令–OS . System()、subprocess . call()–关于日志开发的文章