# Python how to——在 Python 中使用 gzip 模块

> 原文：<https://www.askpython.com/python-modules/gzip-module-in-python>

大家好！在今天的文章中，我们将看看 Python 中的 *gzip* 模块。

这个模块给了我们一个简单的方法来处理 gzip 文件(`.gz`)。这与 Linux 实用程序命令`gzip`和`gunzip`非常相似。

让我们通过一些说明性的例子来看看如何有效地使用这个模块！

* * *

## 在 Python 中使用 gzip 模块

该模块为我们提供了`open()`、`compress()`、`decompress()`等高级函数，用于快速处理这些文件扩展名。

本质上，这将只是打开一个文件！

要导入此模块，您需要以下语句:

```py
import gzip

```

没有必要 [pip 安装](https://www.askpython.com/python-modules/python-pip)这个模块，因为它是标准库的一部分！让我们开始处理一些 gzip 文件。

## 写入压缩文件

我们可以用`gzip.open()`的方法直接打开`.gz`文件，写入这些压缩文件！

```py
import gzip
import os
import io

name = 'sample.txt.gz'

with gzip.open(name, 'wb') as output:
        # We cannot directly write Python objects like strings!
        # We must first convert them into a bytes format using io.BytesIO() and then write it
        with io.TextIOWrapper(output, encoding='utf-8') as encode:
            encode.write('This is a sample text')

# Let's print the updated file stats now
print(f"The file {name} now contains {os.stat(name).st_size} bytes")

```

这里注意，我们不能像写字符串一样直接写 Python 对象！

我们必须首先使用`io.TextIOWrapper()`将它们转换成字节格式，然后使用这个包装函数编写它。这就是为什么我们以二进制写模式(`wb`)打开文件。

如果你运行这个程序，你会得到下面的输出。

**输出**

```py
The file sample.txt.gz now contains 57 bytes

```

此外，您会发现文件`sample.txt.gz`是在当前目录下创建的。好了，我们已经成功地写入了这个压缩文件。

让我们现在尝试解压缩它，并阅读它的内容。

## 从 gzip 文件中读取压缩数据

现在，类似于通过包装的`write()`函数，我们也可以使用相同的函数`read()`。

```py
import gzip
import os
import io

name = 'sample.txt.gz'

with gzip.open(name, 'rb') as ip:
        with io.TextIOWrapper(ip, encoding='utf-8') as decoder:
            # Let's read the content using read()
            content = decoder.read()
            print(content)

```

**输出**

```py
This is a sample text

```

事实上，我们能够得到我们最初写的相同的文本！

## 压缩数据

这个模块的另一个有用的特性是我们可以使用`gzip`有效地压缩数据。

如果我们有很多字节内容作为输入，我们可以使用`gzip.compress()`函数来压缩它。

```py
import gzip

ip = b"This is a large wall of text. This is also from AskPython"
out = gzip.compress(ip)

```

在这种情况下，将使用`gzip.compress`压缩二进制字符串。

* * *

## 结论

在本文中，我们学习了如何使用 Python 中的 gzip 模块来读写`.gz`文件。

## 参考

*   Python gzip 模块[文档](https://docs.python.org/3/library/gzip.html)
*   关于 Python gzip 模块的 JournalDev 文章

* * *