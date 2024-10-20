# 用 Python 散列文件

> 原文：<https://www.pythoncentral.io/hashing-files-with-python/>

请记住，哈希是一个函数，它采用可变长度的字节序列，并将其转换为固定长度的序列。当您需要检查两个文件是否相同，或者需要确保文件的内容没有被更改，以及需要检查文件在网络上传输时的完整性时，计算文件的哈希总是很有用的。有时，当您在网站上下载文件时，网站会提供 MD5 或 SHA 校验和，这很有帮助，因为您可以验证文件是否下载良好。

## 哈希算法

散列文件最常用的算法是 MD5 和 SHA-1。使用它们是因为它们速度快，并且提供了识别不同文件的好方法。哈希函数只使用文件的内容，不使用名称。获得两个独立文件的相同散列意味着文件的内容很可能是相同的，即使它们具有不同的名称。

## Python 中的 MD5 文件哈希

该代码适用于 Python 2.7 和更高版本(包括 Python 3.x)。

```py

import hashlib
hasher = hashlib.md5() 
以 open('myfile.jpg '，' rb ')为 afile:
buf = afile . read()
hasher . update(buf)
print(hasher . hex digest())

```

上面的代码计算文件的 MD5 摘要。文件是以`rb`模式打开的，也就是说你要以二进制模式读取文件。这是因为 MD5 函数需要将文件作为字节序列读取。这将确保您可以散列任何类型的文件，而不仅仅是文本文件。

注意`read`功能很重要。当它在没有参数的情况下被调用时，就像在这种情况下，它将读取文件的所有内容并将它们加载到内存中。如果您不确定文件的大小，这是很危险的。更好的版本是:

## Python 中大型文件的 MD5 哈希

```py

import hashlib

BLOCKSIZE = 65536

hasher = hashlib.md5()

with open('anotherfile.txt', 'rb') as afile:

    buf = afile.read(BLOCKSIZE)

    while len(buf) > 0:

        hasher.update(buf)

        buf = afile.read(BLOCKSIZE)

print(hasher.hexdigest())

```

如果您需要使用另一种算法，只需将`md5`调用更改为另一个支持的函数，例如 SHA1:

## Python 中的 SHA1 文件哈希

```py

import hashlib

BLOCKSIZE = 65536

hasher = hashlib.sha1()

with open('anotherfile.txt', 'rb') as afile:

    buf = afile.read(BLOCKSIZE)

    while len(buf) > 0:

        hasher.update(buf)

        buf = afile.read(BLOCKSIZE)

print(hasher.hexdigest())

```

如果您需要系统中支持的散列算法列表，请使用`hashlib.algorithms_available`。(仅适用于 Python 3.2 及更高版本)。最后，为了更深入地了解散列，请务必查看一下[散列 Python 字符串](https://www.pythoncentral.io/hashing-strings-with-python/ "Hashing Strings with Python")一文。