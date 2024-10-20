# 如何在 Python 中删除文件

> 原文：<https://www.askpython.com/python/delete-files-in-python>

## 介绍

在我们的 Python 文件处理教程中，我们学习了如何在 Python 中操作文件。在本教程中，我们将学习如何在 Python 中删除文件。

在 Python 中，我们知道如何从文件中读取 T1，如何向文件中写入 T3。今天我们来学习一下 Python 中的删除操作。

假设成功创建一个文件后，我们对它执行一些操作，比如读和写。一旦我们完成使用**文件**进行**分析**不同的数据集，也许在某些情况下，我们在未来不需要它。此时**我们如何删除文件？在本教程中，我们将学习。**

## Python 中删除文件的方法

让我们看看在 Python 中删除文件的不同方法。

### 1.使用操作系统模块

**Python** 中的`os`模块提供了一些易于使用的方法，利用这些方法我们可以**删除**或**移除**文件以及**空目录**。仔细查看下面给出的代码:

```py
import os
if os.path.isfile('/Users/test/new_file.txt'):
    os.remove('/Users/test/new_file.txt')
    print("success")
else:    
    print("File doesn't exists!")

```

这里我们使用了一个 **if-else** 语句来避免如果文件目录**不存在**时可能出现的**异常**。方法`isfile()`检查文件名为-**‘new _ file . txt’**的文件是否存在。

同样，`os`模块为我们提供了另一种方法，`rmdir()`，可以用来**删除**或者**删除**一个**空目录**。例如:

```py
import os
os.rmdir('directory')

```

**注意:**目录必须是空的。如果它包含任何内容，方法我们返回一个 **OSerror** 。

### 2.使用 shutil 模块

**shutil** 是 Python 中另一种删除文件的方法，它使得用户可以轻松地**删除文件**或其**完整目录**(包括其所有内容)。

`rmtree()`是 **shutil** 模块下的一个方法，以**递归**的方式删除一个目录及其内容。让我们看看如何使用它:

```py
import shutil
shutil.rmtree('/test/')

```

对于上述代码，删除了目录 **'/test/'** 。**而且最重要的是，目录里面的所有内容也被删除了。**

### 3.使用 pathlib 模块

**pathlib** 是**内置的** python 模块，可供 **Python 3.4+** 使用。我们可以使用这个预定义的模块**删除一个文件**或者一个**空目录**。

让我们举个例子:

```py
import pathlib
file=pathlib.path("test/new_file.txt")
file.unlink()

```

在上面的例子中，`path()`方法用于检索文件路径，而`unlink()`方法用于取消链接或删除指定路径的文件。

unlink()方法适用于文件。如果指定了目录，将引发 OSError。要删除目录，我们可以求助于前面讨论过的方法之一。

## 参考

*   [https://stack overflow . com/questions/6996603/how-to-delete-a-file-or-folder](https://stackoverflow.com/questions/6996603/how-do-i-delete-a-file-or-folder-in-python)
*   [https://docs.python.org/3/library/os.html#os.remove](https://docs.python.org/3/library/os.html#os.remove)
*   [https://docs.python.org/3/library/shutil.html](https://docs.python.org/3/library/shutil.html)
*   [https://docs.python.org/3/library/pathlib.html](https://docs.python.org/3/library/pathlib.html)