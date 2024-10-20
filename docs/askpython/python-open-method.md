# 在 Python 中使用 open()方法打开文件

> 原文：<https://www.askpython.com/python/built-in-methods/python-open-method>

## 介绍

我们已经遇到了使用 Python 可以对文件执行的各种操作，比如[读取](https://www.askpython.com/python/built-in-methods/python-read-file)、[写入](https://www.askpython.com/python/built-in-methods/python-write-file)，或者[复制](https://www.askpython.com/python/copy-a-file-in-python)。在执行任何提到的[文件处理](https://www.askpython.com/python/python-file-handling)操作时，很明显打开文件是第一步。

所以今天在本教程中，我们将重点讨论使用**Python open()方法**打开文件的部分。

## Python 中的 open()方法

`open()`方法以指定的模式打开一个特定的文件，并返回一个**文件对象**。然后，该文件对象可以进一步用于执行各种文件操作。下面给出了使用该方法的语法。

```py
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)

```

这里，

*   **文件**是指文件名/描述符，`mode`是文件打开的模式。这些是打开文件所需的基本参数。
*   **缓冲**是一个可选的整数，用于设置缓冲策略。默认情况下，它被设置为(-1)，
*   **编码**是用于解码或编码文件的编码名称，
*   **errors** 是一个可选字符串，指定如何处理编码和解码错误。注意，这不能在二进制模式下使用。
*   **换行符**控制通用换行符模式的工作方式(仅适用于文本模式)。可以是`None`(默认)`''``'\n'``'\r'``'\r\n'`。
*   **closefd** 表示传递的文件参数是文件名还是文件描述符。当提到文件描述符时，它应该是假的。否则为真(默认)。否则，将会出现错误，
*   **开启器**是一个可调用的定制开启器。通过用(file，flags)调用这个`opener`来获得文件对象的指定文件描述符。opener 必须返回一个打开的文件描述符(传递 [`os.open`](https://docs.python.org/3/library/os.html#os.open) 作为 *opener* 产生类似于传递`None`的功能)。

## Python 中 open()的打开模式

下面给出了不同的文件打开模式及其含义。

| 模式 | 描述 |
| `'r'` | 打开以供阅读(默认) |
| `'w'` | 打开进行写入，首先截断文件 |
| `'x'` | 以独占方式打开，如果文件已经存在，则失败 |
| `'a'` | 打开以供写入，追加到文件的末尾(如果存在) |
| `'b'` | 二进制 |
| `'t'` | 文本模式(默认) |
| `'+'` | 打开以进行更新(读取和写入) |

Table for file opening modes

## Python open()示例

既然我们已经完成了 Python 中`open()`方法的基础，让我们直接进入一些例子。

我们将使用`open()`方法打开一个名为 **file.txt** 的文件，内容如下图所示。

![File Contents](img/00d43523b18664ca07f8e925c532ef9f.png)

File Contents

仔细看看下面给出的代码片段。

```py
# opening a file
f = open('file.txt', 'r')  # file object

print("Type of f: ", type(f))

print("File contents:")

for i in f:
    print(i)

f.close()  # closing file after successful operation

```

**输出:**

```py
Type of f:  <class '_io.TextIOWrapper'>
File contents:
Python

Java

Go

C

C++

Kotlin

```

在这里，我们以只读(`' r '`)模式打开了文件 **file.txt** 。方法将一个文件对象返回给 T2。然后，我们使用循环的[遍历这个对象，以访问文件的内容。](https://www.askpython.com/python/python-for-loop)

之后，我们使用 [close()](https://www.askpython.com/python/python-file-handling#6-close()-function) 方法关闭了文件。在对一个文件执行任何操作后，最后关闭它是很重要的，以避免**错误**。再次打开同一文件时可能会出现这些错误。

## 打开多个文件

在 Python 中，我们可以通过组合使用`with`语句、`open()`方法和逗号(`' , '`)操作符来同时打开两个或多个文件。让我们举一个例子来更好地理解。

这里，我们尝试了打开两个独立的文件 **file1.txt** 和 **file2.txt** 并打印它们对应的内容。

```py
# opening multiple files
try:
    with open('file1.txt', 'r+') as a, open('file2.txt', 'r+') as b:
        print("File 1:")
        for i in a:
            print(i)
        print("File 2:")
        for j in b:
            print(j)
except IOError as e:
    print(f"An Error occured: {e}")

# file closing is not required

```

**输出:**

```py
File 1:
John Alex Leo Mary Jim
File 2:
Sil Rantoff Pard Kim Parsons

```

**注:**本次使用后我们没有关闭文件。因为我们不需要这样做，`with`语句通过调用`close()`方法确保打开的文件自动关闭。

## 结论

今天就到这里吧。希望你有清楚的了解。对于任何进一步的相关问题，请随时使用下面的评论。

我们建议浏览参考资料部分提到的链接以获取更多信息。

## 参考

*   [Python open()](https://docs.python.org/3/library/functions.html#open)–文档，
*   [Python 读取文件——你必须知道的 3 种方式](https://www.askpython.com/python/built-in-methods/python-read-file)，
*   [Python 写文件](https://www.askpython.com/python/built-in-methods/python-write-file)，
*   [用 Python 复制一个文件](https://www.askpython.com/python/copy-a-file-in-python)。