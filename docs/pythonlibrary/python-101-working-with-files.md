# Python 101 -使用文件

> 原文：<https://www.blog.pythonlibrary.org/2020/06/24/python-101-working-with-files/>

应用程序开发人员总是与文件打交道。每当您编写新的脚本或应用程序时，都会创建它们。你用微软 Word 写报告，保存电子邮件或下载书籍或音乐。文件到处都是。您的网络浏览器会下载大量小文件，让您的浏览体验更快。

当您编写程序时，您必须与预先存在的文件交互或自己写出文件。Python 提供了一个很好的内置函数`open()`，可以帮助您完成这些任务。

在本章中，您将学习如何:

*   打开文件
*   读取文件
*   写文件
*   附加到文件

我们开始吧！

### open()函数

您可以打开文件进行读取、写入或附加。要打开一个文件，可以使用内置的`open()`函数。

下面是`open()`函数的参数和默认值:

```py
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, 
     closefd=True, opener=None)
```

当您打开一个文件时，您需要传入一个文件名或文件路径。打开文件的默认方式是以只读模式打开，这就是“r”的意思。

下表介绍了打开文件时可以使用的其他模式:

| 性格；角色；字母 | 意义 |
| --- | --- |
| r ' | 打开文件进行读取(默认) |
| w ' | 打开进行写入，覆盖现有文件 |
| 一个 | 打开进行写入。如果文件存在，追加到末尾 |
| ' b ' | 二进制 |
| t | 文本模式(默认) |
| '+' | 阅读和写作 |

在这一章中，你将着重于阅读、写作和附加。如果您需要将文件编码成特定的格式，比如 UTF-8，您可以通过`encoding`参数来设置。有关 Python 支持的编码类型的完整列表，请参见文档。

有两种打开文件的主要方法。你可以这样做:

```py
file_handler = open('example.txt')
# do something with the file
file_handler.close()
```

在这里你打开文件，然后关闭它。但是，如果在尝试打开文件时出现异常，会发生什么情况呢？例如，假设您试图打开一个不存在的文件。或者您打开了一个文件，但无法写入。这些事情会发生，它们会导致文件句柄处于打开状态而没有正确关闭。

一种解决方案是使用`try/finally`:

```py
try:
    file_handler = open('example.txt')
except:
    # ignore the error, print a warning or log the exception
    pass
finally:
    file_handler.close()
```

然而，用 Python 打开文件的最好方法是使用 Python 特殊的`with`语句。`with`语句激活了所谓的**上下文管理器**。上下文管理器在你想要设置和拆除某些东西的时候使用。在本例中，您想要打开一个文件，执行一些操作，然后关闭该文件。

Python 的核心开发者把`open()`做成一个上下文管理器。这意味着您也可以像这样打开文件:

```py
with open('example.txt') as file_handler:
    # do something with the handler here
    data = file_handler.read()
```

它所做的是打开文件并将文件对象分配给`file_handler`。那么缩进到`with`语句内部的任何代码都被认为是**上下文**的一部分。这是您与文件处理程序交互的地方，无论是读取还是写入文件。然后当你退出`with`语句时，它会自动关闭文件。

就像有一个内置的`finally`语句！

现在你已经知道了如何打开一个文件，让我们继续学习如何用 Python 读取一个文件。

### 读取文件

用 Python 编程语言读取文件非常简单。事实上，当你打开一个文件并且没有设置`mode`参数时，默认是以“只读”模式打开文件。

这里有一个例子:

```py
with open('example.txt') as file_handler:
    for line in file_handler:
        print(line)
```

这段代码将打开文本文件，然后遍历文件中的每一行并打印出来。是的，`file_handler`可以使用 Python 的`for`循环进行迭代，这非常方便。事实上，这实际上是一种推荐的读取文件的方法，因为你是成块读取的，这样你就不会耗尽内存。

另一种循环遍历文件中各行的方法是执行以下操作:

```py
with open('example.txt') as file_handler:
    lines = file_handler.readlines()
    for line in lines:
        print(line)
```

如果你走这条路，那么你只是把整个文件读入内存。根据你的机器有多少内存，你可能会耗尽内存。这就是推荐第一种方法的原因。

但是，如果您知道文件很小，有另一种方法可以将整个文件读入内存:

```py
with open('example.txt') as file_handler:
    file_contents = file_handler.read()
```

方法会把整个文件读入内存，并把它赋给你的变量。

有时，您可能希望以较小或较大的块读取文件。这可以通过将字节大小指定给`read()`来实现。您可以为此使用一个`while`循环:

```py
while True:
    with open('example.txt') as file_handler:
        data = file_handler.read(1024)
        if not data:
            break
        print(data)
```

在本例中，您一次读取 1024 个字节。当您调用`read()`并返回一个空字符串时，那么`while`循环将停止，因为`break`语句将被执行。

### 读取二进制文件

有时你需要读取一个二进制文件。Python 也可以通过将`r`模式与`b`结合起来来实现这一点:

```py
with open('example.pdf', 'rb') as file_handler:
    file_contents = file_handler.read()
```

注意`open()`的第二个参数是`rb`。这告诉 Python 以只读二进制模式打开文件。如果你要打印出`file_contents`，你会看到什么是乱码，因为大多数二进制文件是不可读的。

### 写文件

用 Python 编写一个新文件使用的语法与读取几乎完全相同。但是，对于写入模式，您不是将模式设置为`r`，而是将其设置为`w`。如果你需要用二进制模式写，那么你可以用`wb`模式打开文件。

**警告**:使用`w`和`wb`模式时，如果文件已经存在，最终会被覆盖。Python 不会以任何方式警告你。Python 确实提供了一种通过使用`os`模块经由`os.path.exists()`来检查文件是否存在的方法。更多细节参见 Python 的文档。

让我们向文件中写入一行文本:

```py
>>> with open('example.txt', 'w') as file_handler:
...     file_handler.write('This is a test')
```

这将向文件中写入单行文本。如果你写更多的文字，它将被写在前面的文字旁边。因此，如果您需要添加一个新行，那么您将需要使用`\n`写出一行。

要验证这是否有效，您可以读取该文件并打印出其内容:

```py
>>> with open('example.txt') as file_handler:
...     print(file_handler.read())
... 
This is a test
```

如果需要一次写多行，可以使用`writelines()`方法，该方法接受一系列字符串。例如，你可以使用字符串的`list`并将它们传递给`writelines()`。

### 在文件中查找

文件处理程序还提供了另一个值得一提的方法。这个方法就是`seek()`，你可以用它来改变文件对象的位置。换句话说，您可以告诉 Python 从文件的哪里开始读取。

`seek()`方法接受两个参数:

*   `offset` -来自`whence`的字节数
*   `whence` -参考点

您可以将`whence`设置为以下三个值:

*   0 -文件的开头(默认)
*   1 -当前文件位置
*   2 -文件的结尾

让我们以您在本章前面写的文件为例:

```py
>>> with open('example.txt') as file_handler:
...     file_handler.seek(4)
...     chunk = file_handler.read()
...     print(chunk)
... 
 is a test
```

在这里，您以只读模式打开文件。然后寻找第 4 个字节，将文件的其余部分读入变量`chunk`。最后，打印出`chunk`,看到文件的只读部分。

### 附加到文件

您也可以使用`a`模式将数据追加到预先存在的文件中，这是一种追加模式。

这里有一个例子:

```py
>>> with open('example.txt', 'a') as file_handler:
...     file_handler.write('Here is some more text')
```

如果文件存在，这将在文件末尾添加一个新字符串。另一方面，如果文件不存在，Python 将创建文件并将数据添加到文件中。

### 捕捉文件异常

当您处理文件时，有时会遇到错误。例如，您可能没有创建文件或编辑文件的正确权限。在这种情况下，Python 将引发一个`OSError`。偶尔还会出现其他错误，但这是处理文件时最常见的错误。

您可以使用 Python 的异常处理工具来保持您的程序正常工作:

```py
try:
    with open('example.txt') as file_handler:
        for line in file_handler:
            print(line)
except OSError:
    print('An error has occurred')
```

这段代码将试图打开一个文件并一次打印出一行内容。如果一个`OSError`被引发，你将用`try/except`捕捉它并打印出一条消息给用户。

### 包扎

现在，您已经了解了使用 Python 处理文件的基础知识。在本章中，您学习了如何打开文件。然后你学会了如何读写文件。您还了解了如何在文件中查找、向文件追加内容以及在访问文件时处理异常。在这一点上，你真的只需要实践你所学的。继续尝试你在本章中学到的东西，看看你自己能做什么！