# Python 中文件的行数

> 原文：<https://www.pythonforbeginners.com/basics/number-of-lines-in-a-file-in-python>

文件处理是编程中最重要的操作之一。有时，我们可能需要计算文件中的行数来对其执行任何操作。在本文中，我们将看到如何用 python 计算一个文件中的行数。

## 使用 Python 中的 for 循环计算文件中的行数

计算文件行数的第一种方法是使用 for 循环来计算文件中所有的换行符。

为此，我们将首先在读取模式下使用`open()`功能打开文件。之后，我们将遍历文件内容并检查换行符“`\n`”。我们将在一个名为`numberOfLines`的变量中保存对`\n`字符的计数。

在执行 for 循环后，我们将在变量`numberOfLines`中获得总行数。您可以在下面的示例中观察到这一点。

```py
myFile = open("sample.txt", "r")
text = myFile.read()
print("The content of the file is:")
print(text)
numberOfLines = 0
for character in text:
    if character == "\n":
        numberOfLines = numberOfLines + 1
print("The number of lines in the file is:")
print(numberOfLines) 
```

输出:

```py
The content of the file is:
This is line 1.
This is line 2.
This is line 3.
This is line 4.

The number of lines in the file is:
4
```

在计算行数时，空行也用这种方法计算。这是因为我们正在计算换行符。因此，空行也将被视为新行，因为其中存在“`\n`”字符。

## 使用 Python 中的 split()方法计算文件中的行数

我们可以使用`split()`方法来计算文件中的行数，而不是检查换行符。在字符串上调用 `split()`方法时，该方法将分隔符作为输入参数，并返回原始字符串的子字符串列表。

在我们的程序中，我们将使用“`\n`”作为分隔符，在换行处分割文件的文本。之后，我们将使用`len()`函数确定输出列表的长度。这样，我们就会找到文本文件中的行数。

```py
myFile = open("sample.txt", "r")
text = myFile.read()
print("The content of the file is:")
print(text)
text_list = text.split("\n")
numberOfLines = len(text_list)
print("The number of lines in the file is:")
print(numberOfLines)
```

输出:

```py
The content of the file is:
This is line 1.
This is line 2.
This is line 3.
This is line 4.
The number of lines in the file is:
4
```

## 使用 Python 中的 readlines()方法计算文件中的行数

当在 file 对象上调用时，`readlines()`方法返回文件中的字符串列表。每个字符串由一个换行符组成。我们可以找到输出列表的长度来计算文件中的行数，如下所示。

```py
myFile = open("sample.txt", "r")
text_list = myFile.readlines()
numberOfLines = len(text_list)
print("The number of lines in the file is:")
print(numberOfLines)
```

输出:

```py
The number of lines in the file is:
4
```

## 结论

在本文中，我们讨论了用 python 计算文件行数的三种方法。要了解更多关于文件的内容，您可以阅读这篇关于 python 中的[文件处理的文章。你可能也会喜欢这篇关于如何用 python](https://www.pythonforbeginners.com/filehandling/file-handling-in-python) 让[逐行读取文本文件的文章。](https://www.pythonforbeginners.com/files/4-ways-to-read-a-text-file-line-by-line-in-python)