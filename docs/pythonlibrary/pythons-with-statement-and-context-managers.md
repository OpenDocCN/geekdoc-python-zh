# Python 的语句和上下文管理器

> 原文：<https://www.blog.pythonlibrary.org/2021/04/07/pythons-with-statement-and-context-managers/>

几年前，Python 2.5 中出现了一个特殊的新关键字，即带有和语句的 **。这个新关键字允许开发人员创建上下文管理器。但是等等！什么是上下文管理器？它们是方便的构造，允许你自动设置和拆除某些东西。例如，你可能想打开一个文件，在里面写一些东西，然后关闭它。这可能是上下文管理器的经典例子。事实上，当您使用带有** 语句的 **打开一个文件时，Python 会自动为您创建一个:**

```py
with open(path, 'w') as f_obj:
    f_obj.write(some_data)
```

回到 Python 2.4，您必须用老式的方法来做:

```py
f_obj = open(path, 'w')
f_obj.write(some_data)
f_obj.close()
```

这在幕后的工作方式是通过使用 Python 的一些神奇方法: **__enter__** 和 **__exit__** 。让我们尝试创建您自己的上下文管理器来演示这一切是如何工作的！

## 创建上下文管理器类

这里不用重写 Python 的 open 方法，而是创建一个上下文管理器，它可以创建一个 SQLite 数据库连接，并在连接完成后关闭它。这里有一个简单的例子:

```py
import sqlite3

class DataConn:
    """"""

    def __init__(self, db_name):
        """Constructor"""
        self.db_name = db_name

    def __enter__(self):
        """
        Open the database connection
        """
        self.conn = sqlite3.connect(self.db_name)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the connection
        """
        self.conn.close()
        if exc_val:
            raise

if __name__ == '__main__':
    db = '/home/mdriscoll/test.db'
    with DataConn(db) as conn:
        cursor = conn.cursor()
```

在上面的代码中，您创建了一个获取 SQLite 数据库文件路径的类。 **__enter__** 方法自动执行，创建并返回数据库连接对象。现在您已经有了，您可以创建一个游标并写入数据库或查询它。当您用 语句退出 **时，它会导致 **__exit__** 方法执行并关闭连接。**

让我们尝试使用另一种方法创建一个上下文管理器。

## 使用 contextlib 创建上下文管理器

Python 2.5 不仅用 语句添加了 **，还添加了 **contextlib** 模块。这允许您使用 contextlib 的 **contextmanager** 函数作为装饰器来创建上下文管理器。**

让我们试着创建一个打开和关闭文件的上下文管理器:

```py
from contextlib import contextmanager

@contextmanager
def file_open(path):
    try:
        f_obj = open(path, 'w')
        yield f_obj
    except OSError:
        print("We had an error!")
    finally:
        print('Closing file')
        f_obj.close()

if __name__ == '__main__':
    with file_open('/home/mdriscoll/test.txt') as fobj:
        fobj.write('Testing context managers')
```

这里你从**的 contextlib** 中导入**的 contextmanager** ，并用它来修饰你的 **file_open()** 函数。这允许你使用 Python 的 **和** 语句调用 **file_open** ()。在您的函数中，您打开文件，然后**将它输出**，以便调用函数可以使用它。

一旦带有语句的**结束，控制返回到 **file_open()** ，并继续执行 **yield** 语句之后的代码。这导致执行 **finally** 语句，关闭文件。如果在处理文件时碰巧有一个 **OSError** ，它会被捕获，并且 **finally** 语句仍然会关闭文件处理程序。**

### contextlib.closing()

contextlib 模块附带了一些其他方便的实用程序。第一个是 **closing** 类，它将在代码块完成时关闭这个东西。Python 文档提供了一个类似于以下示例的示例:

```py
from contextlib import contextmanager

@contextmanager
def closing(db):
    try:
        yield db.conn()
    finally:
        db.close()
```

基本上，您所做的是创建一个封装在 **contextmanager** 中的关闭函数。这相当于结束类所做的工作。不同之处在于，您可以在 with 语句中使用**结束**类本身，而不是装饰器。

这将会是这样的:

```py
from contextlib import closing
from urllib.request import urlopen

with closing(urlopen('http://www.google.com')) as webpage:
    for line in webpage:
        # process the line
        pass
```

在这个例子中，你打开一个 URL，但是用你的结束类把它包装起来。这将导致一旦你从**语句的代码块中掉出，网页的句柄就会被关闭。**

 **### context lib . suppress(*异常)

另一个方便的小工具是 Python 3.4 中添加的 **suppress** 类。这个上下文管理器工具背后的思想是它可以抑制任意数量的异常。一个常见的例子是当您想要忽略 **FileNotFoundError** 异常时。如果您要编写以下上下文管理器，它将不起作用:

```py
>>> with open('fauxfile.txt') as fobj:
        for line in fobj:
            print(line)

Traceback (most recent call last):
  Python Shell, prompt 4, line 1
builtins.FileNotFoundError: [Errno 2] No such file or directory: 'fauxfile.txt'
```

这个上下文管理器不处理这个异常。如果您想忽略此错误，则可以执行以下操作:

```py
from contextlib import suppress

with suppress(FileNotFoundError):
    with open('fauxfile.txt') as fobj:
        for line in fobj:
            print(line)
```

在这里，您导入 **suppress** 并向其传递您想要忽略的异常，在本例中是 **FileNotFoundError** 异常。如果运行这段代码，不会发生任何事情，因为文件不存在，但也不会引发错误。应该注意的是，这个上下文管理器是**重入**。这将在本文后面解释。

### contextlib.redirect_stdout / redirect_stderr

contextlib 库有两个用于重定向 stdout 和 stderr 的工具，它们分别是在 Python 3.4 和 3.5 中添加的。在添加这些工具之前，如果您想重定向 stdout，您应该这样做:

```py
path = '/path/to/text.txt'

with open(path, 'w') as fobj:
    sys.stdout = fobj
    help(sum)
```

使用 **contextlib** 模块，您现在可以执行以下操作:

```py
from contextlib import redirect_stdout

path = '/path/to/text.txt'
with open(path, 'w') as fobj:
    with redirect_stdout(fobj):
        help(redirect_stdout)
```

在这两个例子中，您将 stdout 重定向到一个文件。当您调用 Python 的 **help()** 时，它不是打印到 stdout，而是直接保存到文件中。您还可以从 Tkinter 或 wxPython 等用户界面工具包中将 stdout 重定向到某种缓冲区或文本控件类型的小部件。

## 退出堆栈

ExitStack 是一个上下文管理器，它允许你很容易地以编程方式结合其他上下文管理器和清理功能。起初这听起来有点令人困惑，所以让我们来看看 Python 文档中的一个例子，以帮助您更好地理解这个想法:

```py
>>> from contextlib import ExitStack
>>> with ExitStack() as stack:
        file_objects = [stack.enter_context(open(filename))
            for filename in filenames]
                    ]
```

这段代码基本上在 list comprehension 中创建了一系列上下文管理器。 **ExitStack** 维护了一个注册回调的堆栈，当实例关闭时，它会以相反的顺序调用这些回调，这发生在用语句退出**的底部时。**

在 Python 文档中有许多关于 **contextlib** 的简洁示例，您可以从中了解如下主题:

*   从 __enter__ 方法捕获异常

*   支持可变数量的上下文管理器

*   取代任何 try-finally 的使用

*   还有更多！

你应该去看看，这样你就能很好地感受到这个类有多强大。

## 包扎

上下文管理器非常有趣，并且总是很方便。例如，我在自动化测试中一直使用它们来打开和关闭对话框。现在，您应该能够使用 Python 的一些内置工具来创建自己的上下文管理器。请务必花时间阅读 contextlib 上的 Python 文档，因为还有很多本章没有涉及的其他信息。开心快乐编码！**