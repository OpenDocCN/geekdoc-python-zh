# Python 101:如何打开文件或程序

> 原文：<https://www.blog.pythonlibrary.org/2010/09/04/python-101-how-to-open-a-file-or-program/>

当我开始学习 Python 时，我需要知道的第一件事就是打开一个文件。现在，术语“打开文件”可以根据上下文有不同的含义。有时这意味着用 Python 打开文件并从中读取，就像读取文本文件一样。其他时候，它意味着“在默认程序中打开文件”；有时它意味着，“在我指定的程序中打开文件”。所以，当你在寻找如何做到后两者时，你需要知道如何问谷歌正确的问题，否则你最终只能学会如何打开和阅读一个文本文件。

在这篇文章中，我们将涵盖所有三个，我们还将展示如何打开(或运行)程序，已经安装在您的电脑上。为什么？因为这个话题也是我首先需要学习的东西之一，它使用了一些相同的技术。

## 如何打开文本文件

让我们从学习如何用 Python 打开文件开始。在这种情况下，我们的意思是实际使用 Python 打开它，而不是其他程序。为此，我们有两个选择(在 Python 2.x 中):打开或文件。我们来看看，看看是怎么做到的！

```py
# the open keyword opens a file in read-only mode by default
f = open("path/to/file.txt")

# read all the lines in the file and return them in a list
lines = f.readlines()

f.close()
```

如你所见，打开和阅读一个文本文件真的很容易。你可以用“文件”关键字替换“打开”关键字，效果是一样的。如果您想更加明确，您可以这样写 open 命令:

```py
f = open("path/to/file.txt", mode="r")
```

“r”表示只读取文件。也可以用“rb”(读二进制)、“w”(写)、“a”(追加)或“wb”(写二进制)打开文件。请注意，如果您使用“w”或“WB ”, Python 将覆盖该文件(如果它已经存在),或者创建它(如果该文件不存在)。

如果要读取文件，可以使用以下方法:

*   读取整个文件并以字符串的形式返回整个文件
*   **readline** -读取文件的第一行，并将其作为字符串返回
*   读取整个文件并以字符串列表的形式返回

您也可以使用循环读取文件，如下所示:

```py
f = open("path/to/file.txt")
for line in f:
    print line
f.close()
```

很酷吧。巨蟒摇滚！现在是时候看看如何用另一个程序打开一个文件了。

## 用自己的程序打开文件

Python 有一个用默认程序打开文件的简单方法。事情是这样的:

```py
import os
os.startfile(path)
```

是的，很简单，如果你用的是 Windows。如果你在 Unix 或 Mac 上，你将需要子进程模块或“os.system”。当然，如果你是一个真正的极客，那么你可能有多个程序可以用来打开一个特定的文件。例如，我可能想用 Picasa、Paint Shop Pro、Lightroom、Paint 或许多其他程序编辑我的 JPEG 文件，但我不想更改我的默认 JPEG 编辑程序。我们如何用 Python 解决这个问题？我们用的是 Python 的子流程模块！注意:如果你想走老路，你也可以使用 os.popen*或 os.system，但是子进程应该取代它们。

导入子流程

```py
import subprocess

pdf = "path/to/pdf"
acrobat_path = r'C:\Program Files\Adobe\Reader 9.0\Reader\AcroRd32.exe'
subprocess.Popen(f"{acrobat_path} {pdf}")
```

也可以这样写最后一行:*子流程。Popen([acrobatPath，pdf])* 。可以说，使用子流程模块也是轻而易举的事情。使用子流程模块还有许多其他方式，但这是它的主要任务之一。我通常用它来打开一个特定的文件(如上)或打开一个应用了特定参数的程序。我还使用 subprocess 的“call”方法，该方法使 Python 脚本在继续之前等待“被调用”的应用程序完成。如果您知道如何操作，您还可以与子流程启动的流程进行交流。

## 包扎

像往常一样，Python 提供了完成任务的简单方法。我发现 Python 很少不能以一种易于理解的方式雄辩地处理。我希望当你刚开始需要知道如何打开一个文件或程序时，这能对你有所帮助。

## 进一步阅读

*   操作系统模块[文档](http://docs.python.org/library/os.html)
*   子流程模块[文档](http://docs.python.org/library/subprocess.html)
*   [读写文件](http://docs.python.org/tutorial/inputoutput.html#reading-and-writing-files)
*   深入 Python - [使用文件对象](http://diveintopython.org/file_handling/file_objects.html)