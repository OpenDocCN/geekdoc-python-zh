# 用 Python 复制文件

> 原文：<https://www.askpython.com/python/copy-a-file-in-python>

在之前的教程中，我们学习了一些 Python 文件操作，比如[读取](https://www.askpython.com/python/built-in-methods/python-read-file)、[写入](https://www.askpython.com/python/built-in-methods/python-write-file)、[删除](https://www.askpython.com/python/delete-files-in-python)。让我们在本教程中学习用 Python 复制一个文件。

我们可以在下面提到的模块下使用不同的方法在 Python 中复制一个文件，

*   `shutil`模块
*   `os`模块
*   `subprocess`模块

在本教程中，我们将学习使用上述模块提供的不同方法在 Python 中复制文件。

## 1.用 Python 复制文件的 shutil 模块

`shutil`模块提供了一些易于使用的方法，使用这些方法我们可以**删除**以及**复制**Python 中的一个文件。让我们看看在这个模块下定义的专门用于复制的不同方法。

### 1.copyfileobj()

`copyfileobj()`方法使用各自的文件对象将源文件的内容复制到目标文件。让我们看看下面的代码，

```py
import shutil
src_file_obj=open('src.txt', 'rb')
targ_file_obj= open('targ.txt' , 'wb')
shutil.copyfileobj( src_file_obj , targ_file_obj )

```

**注意:**文件对象应该指向各自源文件和目标文件的 **0 位置**(起始位置)，以复制全部内容。

### 2.复制文件()

`copyfile()`方法使用文件路径将内容从源复制到目标文件。它返回目标文件路径。目标文件路径必须是可写的，否则会发生 **OSerror** 异常。

```py
import shutil
shutil.copyfile( 'src.txt' , 'targ.txt' )

```

请记住，该方法只允许使用文件路径，而不允许使用目录。

### 3.复制()

此方法将源文件复制到目标文件或目标目录。与`copyfile()`不同，方法`copy()`允许使用目标目录作为参数，并且还复制文件权限。`copy()`复制内容后返回目标文件的路径。

```py
import shutil
shutil.copy('/Users/test/file.txt', '/Users/target/')

```

在目标位置创建一个名为 **'file.txt'** 的文件，其中所有内容和权限都是从 **'/Users/test/file.txt '中复制的。**

### 4.副本 2()

`copy2()`方法的使用方式与`copy()`方法完全相同。它们也以同样的方式工作，除了，因为`copy2()`也从源文件中复制**元数据**。

```py
import shutil
shutil.copy2('/Users/test/file.txt', '/Users/target/')

```

## 2.用 Python 复制文件的操作系统模块

### 1.波本()

方法`popen()`创建一个到命令 **cmd** 的管道。该方法返回一个连接到 cmd 管道的文件对象。看看下面的代码，

```py
#for Windows
import os
os.popen('copy src.txt targ.txt' )

```

```py
#for Linux
import os
os.popen('cp src.txt targ.txt' )

```

用这种方法，我们不仅可以复制文件，还可以执行其他常规命令。

### 2.系统()

`system()`方法直接调用并执行 subshell 中的命令参数。它的返回值取决于运行该程序的操作系统。对于 Linux，它是退出状态，而对于 Windows，它是系统 shell 的返回值。

```py
#for Linux
import os
os.system(' cp src.txt targ.txt' )

```

```py
#for Windows
import os
os.system(' copy src.txt targ.txt' )

```

## 3.用 Python 复制文件的子过程模块

### 1.调用()

类似于`os.system()`的`call()`方法直接调用或运行作为参数传递给函数的命令。

```py
# In Linux
import subprocess
subprocess.call('cp source.txt target.txt', shell=True)

```

```py
# In Windows
import subprocess
subprocess.call('copy source.txt target.txt', shell=True)

```

## 参考

*   [https://docs . python . org/3/library/subprocess . html # subprocess . call](https://docs.python.org/3/library/subprocess.html#subprocess.call)
*   [https://docs.python.org/2/library/os.html](https://docs.python.org/2/library/os.html)
*   [https://docs.python.org/3/library/shutil.html](https://docs.python.org/3/library/shutil.html)
*   [https://stack overflow . com/questions/123198/how-do-I-copy-a-file-in-python](https://stackoverflow.com/questions/123198/how-to-copy-files)