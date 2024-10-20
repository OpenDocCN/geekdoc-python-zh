# Python how to–在 Python 中使用 tempfile 模块

> 原文：<https://www.askpython.com/python/tempfile-module-in-python>

大家好！在今天的文章中，我们将看看如何在 Python 中使用 tempfile 模块。

当你想存储临时文件时，这个模块非常有用。从应用程序的角度来看，可能需要存储临时数据，因此这些文件可能非常有用！

Python 为我们提供了`tempfile`模块，这给了我们一个易于使用的接口。让我们开始吧。

* * *

## Python 中的 tempfile 模块

这个模块是标准库(Python 3.x)的一部分，所以你不需要用 [pip](https://www.askpython.com/python-modules/python-pip) 安装任何东西。可以简单导入！

```py
import tempfile

```

我们现在将看看如何创建临时文件和目录。

### 创建临时文件和目录

`tempfile`模块给了我们`TemporaryFile()`方法，它将创建一个临时文件。

由于该文件是临时的，其他程序 ***不能*** 直接访问该文件。

作为一般的安全措施，Python 会在关闭后自动删除任何创建的临时文件。即使它保持打开，在我们的程序完成后，这个临时数据将被删除。

现在让我们看一个简单的例子。

```py
import tempfile

# We create a temporary file using tempfile.TemporaryFile()
temp = tempfile.TemporaryFile()

# Temporary files are stored here
temp_dir = tempfile.gettempdir()

print(f"Temporary files are stored at: {temp_dir}")

print(f"Created a tempfile object: {temp}")
print(f"The name of the temp file is: {temp.name}")

# This will clean up the file and delete it automatically
temp.close()

```

**输出**

```py
Temporary files are stored at: /tmp
Created a tempfile object: <_io.BufferedRandom name=3>
The name of the temp file is: 3

```

现在让我们试着找到这个文件，使用`tempfile.gettempdir()`获取存储所有临时文件的目录。

运行完程序后，如果您转到`temp_dir`(在我的例子中是`/tmp`——Linux ),您可以看到新创建的文件`3`不在那里。

```py
ls: cannot access '3': No such file or directory

```

这证明 Python 在这些临时文件关闭后会自动删除它们。

现在，类似于创建临时文件，我们也可以使用`tempfile.TemporaryDirectory()`函数创建临时目录。

```py
tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=None)

```

目录名是随机的，所以您可以指定一个可选的`suffix`和/或`prefix`来标识它们，作为您程序的一部分。

同样，为了确保在相关代码完成后安全删除目录，我们可以使用一个[上下文管理器](https://www.askpython.com/python/python-with-context-managers)来安全地包装它！

```py
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    # The context manager will automatically delete this directory after this section
    print(f"Created a temporary directory: {tmpdir}")

print("The temporary directory is deleted")

```

**输出**

```py
Created a temporary directory: /tmp/tmpa3udfwu6
The temporary directory is deleted

```

还是那句话，要验证这一点，你可以试着去相关的目录路径，不会存在的！

### 1.从临时文件中读写

类似于从文件中读取或写入，我们也可以从临时文件中使用相同类型的函数调用来做到这一点！

```py
import tempfile

with tempfile.TemporaryFile() as fp:
    name = fp.name
    fp.write(b'Hello from AskPython!') # Write a byte string using fp.write()
    fp.seek(0) # Go to the start of the file
    content = fp.read() # Read the contents using fp.read()
    print(f"Content of file {name}: {content}")

print("File is now deleted")

```

现在让我们看看输出。

**输出**

```py
Content of file 3: b'Hello from AskPython!'
File is now deleted

```

事实上，我们也能够轻松地读写临时文件。

### 2.创建命名的临时文件

在某些情况下，命名的临时文件可能有助于使文件对其他脚本/进程可见，以便它们可以在它尚未关闭时访问它。

`tempfile.NamedTemporaryFile()`对此很有用。这与创建普通临时文件的语法相同。

```py
import tempfile

# We create a named temporary file using tempfile.NamedTemporaryFile()
temp = tempfile.NamedTemporaryFile(suffix='_temp', prefix='askpython_')

print(f"Created a Named Temporary File {temp.name}")

temp.close()

print("File is deleted")

```

**输出**

```py
Created a Named Temporary File /tmp/askpython_r2m23q4x_temp
File is deleted

```

这里，创建了一个带有前缀`askpython_`和后缀`_temp`的命名临时文件。同样，关闭后会自动删除。

* * *

## 结论

在本文中，我们学习了如何使用 Python 中的 tempfile 模块来处理临时文件和目录。

## 参考

*   Python 临时文件模块[文档](https://docs.python.org/3/library/tempfile.html)
*   tempfile 模块上的 JournalDev 文章

* * *