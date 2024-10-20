# Oshash 模块快速介绍

> 原文：<https://www.askpython.com/python-modules/oshash-module>

大家好！oshash 模块将在今天的教程中讨论。我们将探索如何将它融入我们的系统并加以利用。我们还将分析该方法与其他算法在性能方面的比较。接下来，我们将查看它的一些实例，以便更好地理解它。

那么，我们开始吧，好吗？

* * *

## 哈希简介

**散列**是使用函数或算法将对象数据映射到代表性整数值的过程。这是通过使用具有键值对的表来实现的。它通过散列函数绕过该值进行操作，散列函数返回一个与该值对应的密钥，也称为**散列密钥/散列码**。然后，整数哈希代码被映射到我们拥有的固定大小。

由此我们可以得出结论，散列函数是可以用来将可变大小的数据转换成固定大小的值的任何函数。哈希值、哈希代码或简单的哈希是哈希函数返回的值。现在我们已经对散列有了基本的了解，我们可以继续学习模块" **oshash** "

* * *

## whatmakesosashmodulebetter

虽然有各种有效的算法，但“T0”Oshash 探索了一些不同的技术来实现散列。与其他算法相比，它的主要目的是在其他算法落后时获得好的速度。

使它们变得迟缓的主要缺点是它们一次读取整个文件，这对于“oshash”是不推荐的。相反，它逐段读取文件。

然而，我们不必担心它的内部操作或散列函数。我们将更多地关注它的应用。让我们从安装开始，然后再看例子。

### 安装 Oshash 模块

我们可以使用 pip 和以下命令来安装它。

```py
pip install oshash

```

* * *

## 实施 Oshash 模块

所以，一旦我们完成了安装，让我们看看如何把它投入使用。

我们可以通过两种方式利用它:第一种是在我们的**程序文件**中，第二种是通过**命令行界面**。让我们来看一个例子。在这两种情况下，它都返回一个散列文件。

### 程序文件的语法

```py
import oshash
file_hash = oshash.oshash(<path to video file>)

```

### 命令行界面的语法

```py
$ oshash <path to file>

```

虽然我们在前面的例子中没有看到任何这样的技术，但是在后台创建了一个散列，如下面的语法所示。

```py
file_buffer = open("/path/to/file/")

head_checksum = checksum(file_buffer.head(64 * 1024))  # 64KB
tail_checksum = checksum(file_buffer.tail(64 * 1024))  # 64KB

file_hash = file_buffer.size + head_checksum + tail_checksum

```

* * *

## 结论

恭喜你！您刚刚学习了 Python 中的 Oshash 模块。希望你喜欢它！😇

喜欢这个教程吗？无论如何，我建议你看一下下面提到的教程:

1.  [xlrd 模块——如何在 Python 中处理 Excel 文件？](https://www.askpython.com/python-modules/xlrd-module)
2.  [pyzbar 模块:用 Python 解码条形码](https://www.askpython.com/python-modules/pyzbar-module)
3.  [Python HTTP 模块–您需要知道的一切！](https://www.askpython.com/python-modules/http-module)
4.  [Python 制表模块:如何在 Python 中轻松创建表格？](https://www.askpython.com/python-modules/tabulate-tables-in-python)

感谢您抽出时间！希望你学到了新的东西！！😄

* * *