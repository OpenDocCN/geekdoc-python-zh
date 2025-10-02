# 第十四章 Python 标准库

**目录表**

*   简介
*   sys 模块
*   命令行参数
*   更多 sys 的内容
*   os 模块
*   概括

Python 标准库是随 Python 附带安装的，它包含大量极其有用的模块。熟悉 Python 标准库是十分重要的，因为如果你熟悉这些库中的模块，那么你的大多数问题都可以简单快捷地使用它们来解决。

我们已经研究了一些这个库中的常用模块。你可以在 Python 附带安装的文档的“库参考”一节中了解 Python 标准库中所有模块的完整内容。

# sys 模块

`sys`模块包含系统对应的功能。我们已经学习了`sys.argv`列表，它包含命令行参数。

```py
#!/usr/bin/python
# Filename: cat.py

import sys

def readfile(filename):
    '''Print a file to the standard output.'''
    f = file(filename)
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        print line, # notice comma
    f.close()

# Script starts from here
if len(sys.argv) &lt; 2:
    print 'No action specified.'
    sys.exit()

if sys.argv[1].startswith('--'):
    option = sys.argv[1][2:]
    # fetch sys.argv[1] but without the first two characters
    if option == 'version':
        print 'Version 1.2'
    elif option == 'help':
        print '''\
This program prints files to the standard output.
Any number of files can be specified.
Options include:
  --version : Prints the version number
  --help    : Display this help'''
    else:
        print 'Unknown option.'
    sys.exit()
else:
    for filename in sys.argv[1:]:
        readfile(filename) 
```

（源文件：code/cat.py）

## 输出

```py
$ python cat.py
No action specified.

$ python cat.py --help
This program prints files to the standard output.
Any number of files can be specified.
Options include:
  --version : Prints the version number
  --help    : Display this help

$ python cat.py --version
Version 1.2

$ python cat.py --nonsense
Unknown option.

$ python cat.py poem.txt
Programming is fun
When the work is done
if you wanna make your work also fun:
        use Python! 
```

## 它如何工作

这个程序用来模范 Linux/Unix 用户熟悉的**cat**命令。你只需要指明某些文本文件的名字，这个程序会把它们打印输出。

在 Python 程序运行的时候，即不是在交互模式下，在`sys.argv`列表中总是至少有一个项目。它就是当前运行的程序名称，作为`sys.argv[0]`（由于 Python 从`0`开始计数）。其他的命令行参数在这个项目之后。

为了使这个程序对用户更加友好，我们提供了一些用户可以指定的选项来了解更多程序的内容。我们使用第一个参数来检验我们的程序是否被指定了选项。如果使用了`--version`选项，程序的版本号将被打印出来。类似地，如果指定了`--help`选项，我们提供一些关于程序的解释。我们使用`sys.exit`函数退出正在运行的程序。和以往一样，你可以看一下`help(sys.exit)`来了解更多详情。

如果没有指定任何选项，而是为程序提供文件名的话，它就简单地打印出每个文件地每一行，按照命令行中的顺序一个文件接着一个文件地打印。

顺便说一下，名称**cat**是 concatenate 的缩写，它基本上表明了程序的功能——它可以在输出打印一个文件或者把两个或两个以上文件连接/级连在一起打印。

`sys.version`字符串给你提供安装的 Python 的版本信息。`sys.version_info`元组则提供一个更简单的方法来使你的程序具备 Python 版本要求功能。

```py
[swaroop@localhost code]$ python
>>> import sys
>>> sys.version
'2.3.4 (#1, Oct 26 2004, 16:42:40) \n[GCC 3.4.2 20041017 (Red Hat 3.4.2-6.fc3)]'
>>> sys.version_info
(2, 3, 4, 'final', 0) 
```

对于有经验的程序员，`sys`模块中其他令人感兴趣的项目有`sys.stdin`、`sys.stdout`和`sys.stderr`它们分别对应你的程序的标准输入、标准输出和标准错误流。

# os 模块

这个模块包含普遍的操作系统功能。如果你希望你的程序能够与平台无关的话，这个模块是尤为重要的。即它允许一个程序在编写后不需要任何改动，也不会发生任何问题，就可以在 Linux 和 Windows 下运行。一个例子就是使用`os.sep`可以取代操作系统特定的路径分割符。

下面列出了一些在`os`模块中比较有用的部分。它们中的大多数都简单明了。

*   `os.name`字符串指示你正在使用的平台。比如对于 Windows，它是`'nt'`，而对于 Linux/Unix 用户，它是`'posix'`。

*   `os.getcwd()`函数得到当前工作目录，即当前 Python 脚本工作的目录路径。

*   `os.getenv()`和`os.putenv()`函数分别用来读取和设置环境变量。

*   `os.listdir()`返回指定目录下的所有文件和目录名。

*   `os.remove()`函数用来删除一个文件。

*   `os.system()`函数用来运行 shell 命令。

*   `os.linesep`字符串给出当前平台使用的行终止符。例如，Windows 使用`'\r\n'`，Linux 使用`'\n'`而 Mac 使用`'\r'`。

*   `os.path.split()`函数返回一个路径的目录名和文件名。

    ```py
    >>> os.path.split('/home/swaroop/byte/code/poem.txt')
    ('/home/swaroop/byte/code', 'poem.txt') 
    ```

*   `os.path.isfile()`和`os.path.isdir()`函数分别检验给出的路径是一个文件还是目录。类似地，`os.path.existe()`函数用来检验给出的路径是否真地存在。

你可以利用 Python 标准文档去探索更多有关这些函数和变量的详细知识。你也可以使用`help(sys)`等等。

# 概括

我们已经学习了 Python 标准库中的`sys`模块和`os`模块的一部分功能。你应该利用 Python 标准文档去学习这两个模块以及其他模块的更多内容。

接下来，我们将要学习 Python 中剩余的几个方面的内容，从而使我们的 Python 课程更加 完整 。