# 用 Python 向文件追加文本

> 原文：<https://www.pythonforbeginners.com/basics/append-text-to-file-in-python>

在进行文件操作时，我们可能需要在不删除现有数据的情况下向现有文件添加文本。在本文中，我们将讨论如何在 python 中向文件追加文本。

## 使用 write()方法将文本追加到文件中

要使用`write()`方法将文本添加到文件中，我们首先需要在 append 模式下打开文件。为此，我们将使用`open()`函数，将文件名作为第一个参数，将“`r+`”作为第二个参数。打开文件后，我们可以使用`write()`方法简单地将文本添加到文件中。在 file 对象上调用`write()`方法，并将需要追加到文件中的文本作为其输入参数。你可以在下面观察整个过程。

```py
myFile = open("sample.txt", mode="r+")
print("The content of the file before modification is:")
text = myFile.read()
print(text)
myString = "This string will be appended to the file."
myFile.write(myString)
myFile.close()
myFile = open("sample.txt", "r")
print("The content of the file after modification is:")
text = myFile.read()
print(text)
```

输出:

```py
The content of the file before modification is:
This is a sample file.
The content of the file after modification is:
This is a sample file.This string will be appended to the file.
```

将文本附加到文件后，不要忘记关闭文件。否则，内容将不会被保存。这里，我们使用了 read()函数来验证附加文本前后的文件内容。

## 使用 print()函数将文本添加到文件中

通常，当我们使用`print()`函数时，它会将值打印到标准输入中。然而，我们也可以使用`print()`函数在 python 中将文本追加到文件中。`print()`函数有一个可选参数“`file`”。使用这个参数，我们可以指定在哪里打印作为输入传递给`print()`函数的值。

为了将文本追加到文件中，我们将首先使用`open()`函数在追加模式下打开文件。之后，我们将把文本和文件对象分别作为第一个和第二个输入参数传递给打印函数。执行`print()`功能后，文本将被附加到文件中。

```py
myFile = open("sample.txt", mode="r+")
print("The content of the file before modification is:")
text = myFile.read()
print(text)
myString = "This string will be appended to the file."
print(myString, file=myFile)
myFile.close()
myFile = open("sample.txt", "r")
print("The content of the file after modification is:")
text = myFile.read()
print(text)
```

输出:

```py
The content of the file before modification is:
This is a sample file.

The content of the file after modification is:
This is a sample file.
This string will be appended to the file.
```

在输出中，您可以观察到`myString`已经被追加到文件的新行中。当我们使用`write()`方法做同样的操作时，`myString`被附加到现有文件的最后一行。因此，您可以利用这种差异根据您的需求选择合适的方法。此外，请确保在向文件追加文本后关闭该文件。否则，将不会保存更改。

## 结论

在本文中，我们讨论了用 python 向文件追加文本的两种方法。要了解更多关于文件操作的信息，您可以阅读这篇关于 python 中的[文件处理的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/filehandling/file-handling-in-python) 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)