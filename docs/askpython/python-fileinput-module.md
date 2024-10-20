# Python how to–使用 Python 文件输入模块

> 原文：<https://www.askpython.com/python-modules/python-fileinput-module>

大家好！在本文中，我们将了解如何使用 Python fileinput 模块。这是一个非常方便的实用模块，可以快速浏览作为输入的文件列表。

让我们看看如何使用这个模块有效地遍历输入文件。

* * *

## 如何使用 Python 文件输入模块

这是 Python 标准库的一部分，所以不需要 [pip 安装](https://www.askpython.com/python-modules/python-pip)这个模块。

为了导入这个模块，我们可以使用下面的语句:

```py
import fileinput

```

通常，如果你想对单个输入文件做一些 IO 操作(读/写)，我们一般使用 [open()函数](https://www.askpython.com/python/built-in-methods/open-files-in-python)来实现。

但是，如果您需要传递多个文件，我们可以使用`fileinput`直接快速遍历所有文件。

现在我们来看一个例子。

### 1.读取多个文件

这个模块的主要用途是将`fileinput.FileInput`实例用作[上下文管理器](https://www.askpython.com/python/python-with-context-managers)。

```py
import fileinput

with fileinput.FileInput(files=('a.txt', 'b.txt'), mode='r') as input:
    ...

```

在这里，我们可以向关键字参数`files`传递任意多的文件。也允许使用单个文件。

要指定打开文件的模式，我们必须指定关键字参数`mode`。

假设我们的目录有以下两个文件`a.txt`和`b.txt`，内容如下:

```py
$ cat a.txt
Hello from AskPython!
This is a.txt

$ cat b.txt
Hello from AskPython!
this is b.txt

```

现在，我们将把这两个文件作为输入传递给我们的示例程序:

```py
import fileinput

with fileinput.FileInput(files=('a.txt', 'b.txt'), mode='r') as input:
    for line in input:
        print(line)

```

**输出**

```py
Hello from AskPython!

This is a.txt

Hello from AskPython!

This is b.txt

```

事实上，我们能够打印这两个文件！每行之间的空格是由于`print()`在每个语句后添加了一个新行。由于我们的文件已经有了新的行，它将在中间打印一个额外的行。

### 2.验证第一行并读取文件名

现在，我们可以利用这个模块的其他方法。

如果您想查看当前正在读取的文件的名称，我们可以使用`fileinput.filename()`方法。

然而，如果没有行被读取，这将返回`None`！所以你只能在第一次阅读后使用。

如果我们想找出正在读取的文件的名称，我们可以再使用一个标志。

如果读取的行是第一行，`fileinput.isfirstline()`方法将返回`True`！因此，如果该标志为真，我们可以打印到控制台。

这里有一个简单的例子，对`a.txt`和`b.txt`使用相同的程序

```py
import fileinput

with fileinput.FileInput(files=('a.txt', 'b.txt'), mode='r') as input:
    for idx, line in enumerate(input):
        if input.isfirstline() == True:
            # We will indicate the file name being read if the first line is read
            print(f'Reading file {input.filename()}...')
        print(line)

```

**输出**

```py
Reading file a.txt...
Hello from AskPython!

This is a.txt

Reading file b.txt...
Hello from AskPython!

This is b.txt

```

正如你所看到的，当第一行是 beign read 时，我们能够查看正在读取的文件的名称。

类似地，我们可以使用其他助手方法快速地在输入文件中迭代。

要了解更多，您可以查看[文档](https://docs.python.org/3/library/fileinput.html#fileinput.filename)。

* * *

## 结论

在本文中，我们学习了如何使用 Python 中的 fileinput 模块快速遍历来自`stdin`的输入文件。

## 参考

*   文件输入模块[文档](https://docs.python.org/3/library/fileinput.html)

* * *