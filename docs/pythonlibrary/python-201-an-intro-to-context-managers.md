# Python 201 -上下文管理器简介

> 原文：<https://www.blog.pythonlibrary.org/2015/10/20/python-201-an-intro-to-context-managers/>

几年前，Python 2.5 中出现了一个特殊的新关键字，称为“ **with** statement”。这个新关键字允许开发人员创建上下文管理器。但是等等！什么是上下文管理器？它们是方便的构造，允许你自动设置和拆除某些东西。例如，你可能想打开一个文件，在里面写一些东西，然后关闭它。这可能是上下文管理器的经典示例:

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

这在幕后的工作方式是通过使用 Python 的一些神奇方法: **__enter__** 和 **__exit__** 。让我们试着创建我们自己的上下文管理器来演示这一切是如何工作的！

### 创建上下文管理器类

这里我们不重写 Python 的 open 方法，而是创建一个上下文管理器，它可以创建一个 SQLite 数据库连接，并在连接完成后关闭它。这里有一个简单的例子:

```py

import sqlite3

########################################################################
class DataConn:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, db_name):
        """Constructor"""
        self.db_name = db_name

    #----------------------------------------------------------------------
    def __enter__(self):
        """
        Open the database connection
        """
        self.conn = sqlite3.connect(self.db_name)
        return self.conn

    #----------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Close the connection
        """
        self.conn.close()

#----------------------------------------------------------------------
if __name__ == '__main__':
    db = '/home/mdriscoll/test.db'
    with DataConn(db) as conn:
        cursor = conn.cursor()

```

在上面的代码中，我们创建了一个获取 SQLite 数据库文件路径的类。 **__enter__** 方法自动执行，创建并返回数据库连接对象。现在我们有了它，我们可以创建一个游标并写入数据库或查询它。当我们用语句退出**时，会导致 **__exit__** 方法执行并关闭连接。**

让我们尝试使用另一种方法创建一个上下文管理器。

* * *

### 使用 contextlib 创建上下文管理器

Python 2.5 不仅增加了带语句的**，还增加了**上下文库**模块。这允许我们使用 contextlib 的 **contextmanager** 函数作为装饰器来创建上下文管理器。让我们试着创建一个打开和关闭文件的上下文管理器:**

```py

from contextlib import contextmanager

@contextmanager
def file_open(path):
    try:
        f_obj = open(path, 'w')
        yield f_obj
    except OSError:
        print "We had an error!"
    finally:
        print 'Closing file'
        f_obj.close()

#----------------------------------------------------------------------
if __name__ == '__main__':
    with file_open('/home/mdriscoll/test.txt') as fobj:
        fobj.write('Testing context managers')

```

这里我们只是从**的 contextlib** 中导入 **contextmanager** ，并用它来修饰我们的 **file_open** 函数。这允许我们使用 Python 的 **with** 语句调用 **file_open** 。在我们的函数中，我们打开文件，然后**将它输出**，以便调用函数可以使用它。一旦 with 语句结束，控制返回到 **file_open** 函数，并继续执行 **yield** 语句后面的代码。这导致执行 **finally** 语句，关闭文件。如果我们在处理文件时碰巧有一个 **OSError** ，它会被捕获，并且**最终**语句仍然会关闭文件处理程序。

* * *

### 包扎

上下文管理器非常有趣，并且总是很方便。例如，我在自动化测试中一直使用它们来打开和关闭对话框。现在，您应该能够使用 Python 的一些内置工具来创建自己的上下文管理器。开心快乐编码！

* * *

### 相关阅读

*   上下文库[文档](https://docs.python.org/2/library/contextlib.html)
*   [理解 Python 的“with”语句](http://effbot.org/zone/python-with-statement.htm)