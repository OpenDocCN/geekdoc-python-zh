# 用 Python 逐行读取文件

> 原文：<https://www.pythonforbeginners.com/basics/read-file-line-by-line-in-python>

文件操作在各种任务中至关重要。在本文中，我们将讨论如何在 python 中逐行读取文件。

## 使用 readline()方法读取文件

Python 为我们提供了`readline()`方法来[读取文件](https://www.pythonforbeginners.com/files/4-ways-to-read-a-text-file-line-by-line-in-python)。要读取文件，我们将首先在读取模式下使用`open()`功能打开文件。`open()`函数将文件名作为第一个输入参数，将文字“r”作为第二个输入参数，表示文件以读取模式打开。执行后，它返回包含该文件的 file 对象。

得到 file 对象后，我们可以使用`readline()`方法来读取文件。在 file 对象上调用`readline()`方法时，返回文件中当前未读的行，并将迭代器移动到文件中的下一行。

为了逐行读取文件，我们将使用 `readline()`方法读取文件中的每一行，并在 while 循环中打印出来。一旦`readline()`方法到达文件的末尾，它将返回一个空字符串。因此，在 while 循环中，我们还将检查从文件中读取的内容是否为空字符串，如果是，我们将从 for 循环中退出。

使用`readline()`方法读取文件的 python 程序如下。

```py
myFile = open('sample.txt', 'r')
print("The content of the file is:")
while True:
    text = myFile.readline()
    if text == "":
        break
    print(text, end="")
myFile.close()
```

输出:

```py
The content of the file is:
I am a sample text file.
I was created by Aditya.
You are reading me at Pythonforbeginners.com.
```

建议文章:[使用 C# |点网核心| SSH.NET 将文件上传到 SFTP 服务器](https://codinginfinite.com/upload-file-sftp-server-using-csharp-net-core-ssh/)

## 使用 readlines()方法在 Python 中逐行读取文件

在 python 中，我们可以使用`readlines()` 方法来读取文件，而不是使用 `readline()`方法。当在 file 对象上调用`readlines()` 方法时，它返回一个字符串列表，其中列表中的每个元素都是文件中的一行。

打开文件后，我们可以使用`readlines()`方法获得文件中所有行的列表。之后，我们可以使用一个 for 循环来逐个打印文件中的所有行，如下所示。

```py
myFile = open('sample.txt', 'r')
print("The content of the file is:")
lines = myFile.readlines()
for text in lines:
    print(text, end="")
myFile.close()
```

输出:

```py
The content of the file is:
I am a sample text file.
I was created by Aditya.
You are reading me at Pythonforbeginners.com.
```

## 结论

在本文中，我们讨论了在 python 中逐行读取文件的两种方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于 Python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。