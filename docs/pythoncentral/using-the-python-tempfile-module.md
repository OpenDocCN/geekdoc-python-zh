# 使用 Python 临时文件模块

> 原文：<https://www.pythoncentral.io/using-the-python-tempfile-module/>

在用 Python 编程时，可能会有这样的时候，您需要以文件的形式使用或操作一些数据，但是还没有写入文件。自然地，想到的第一个解决方案是打开一个新的或现有的文件，写入数据并最终保存它(如果您不熟悉如何做，可以看看文章[在 Python 中读取和写入文件](https://www.pythoncentral.io/reading-and-writing-to-files-in-python/))。然而，也可能出现这样的情况，一旦脚本运行完毕，您就不再需要或想要这些文件了，因此，不希望它留在您或其他任何人的文件系统中。

这就是 *tempfile* 模块派上用场的地方，它提供了创建临时文件的函数，当脚本退出时，您不必去手动删除这些文件。让我们看几个简单的例子来说明使用*临时文件*的基本方法。

用 *tempfile* 创建一个临时文件就像用内置的 open()方法创建一个常规文件一样，只是不需要给临时文件命名。我们将打开一个，向其中写入一些数据，然后读回我们所写的内容。

```py

# Importing only what we'll use

from tempfile import TemporaryFile

t = TemporaryFile()

data = 'A simple string of text.'

```

现在我们有了要写入的临时文件 *t* 和*数据*字符串——我们没有指定文件的模式，所以它默认为‘w+b ’,这意味着我们可以读取和写入任何类型的数据。在接下来的三行中，我们将把数据写入文件，然后读回我们写的内容。

*   [Python 3.x](#custom-tab-0-python-3-x)
*   [Python 2.x](#custom-tab-0-python-2-x)

*   [Python 3.x](#)

[python]
>>> t.write(bytes(data, 'UTF-8'))
>>> # Makes sure the marker is at the start of the file
>>> t.seek(0)
>>> print(t.read().decode())
A simple string of text.
[/python]

*   [Python 2.x](#)

[python]
>>> t.write(data)
>>> # Makes sure the marker is at the start of the file
>>> t.seek(0)
>>> print(t.read())
A simple string of text.
[/python]

就像其他文件对象一样，我们可以向临时文件对象写入数据，也可以从临时文件对象中读取数据。同样，区别和优点是，一旦临时文件对象被关闭，就没有它的踪迹了。这在使用 with 语句时尤其有利，该语句会在语句完成时自动执行关闭文件的简单清理工作:

*   [Python 3.x](#custom-tab-1-python-3-x)
*   [Python 2.x](#custom-tab-1-python-2-x)

*   [Python 3.x](#)

[python]
with TemporaryFile() as tempf:
data = (data + '\n') * 3
# Write the string thrice
tempf.write(bytes(data, 'UTF-8'))
tempf.seek(0)
print(tempf.read().decode())
[/python]

*   [Python 2.x](#)

[python]
with TemporaryFile() as tempf:
# Write the string thrice
tempf.write((data + '\n') * 3)
tempf.seek(0)
print(tempf.read())
[/python]

我们得到以下输出:

```py

A simple string of text.

A simple string of text.

A simple string of text.

```

并检查文件句柄是否已关闭:

```py

>>> # Test to see if the file has been closed

>>> print(tempf.closed)

True

```

这基本上是临时文件的要点。它和其他文件对象一样简单易用。

*tempfile* 模块还提供了 NamedTemporaryFile()方法，该方法提供了一个临时文件，该文件在文件系统中总是有一个明确且可见的名称(当然，直到它被关闭)。文件名可以通过它的 name 属性来访问(如果我们的文件 t 是一个 NamedTemporaryFile，我们将使用 t.name 来访问该信息)。此外，这种类型的临时文件提供了在关闭时实际保存文件而不是删除文件的选项。

如果你需要创建一个临时目录，这个模块也提供了 mkdtemp()方法来实现；然而，与 TemporaryFile 和 NamedTemporaryFile 对象不同，临时目录 **不会在没有您手动删除它们的情况下被** 删除——它们唯一的临时之处是，它们默认存储在由 tempfile.tempdir 的值定义的临时文件夹中。尽管如此， *tempfile* 提供的创建临时文件和目录的工具还是非常有用的，所以不要犹豫尝试一下。