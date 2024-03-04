# 将 CSV 读入 Python 中的列表列表

> 原文：<https://www.pythonforbeginners.com/basics/read-csv-into-a-list-of-lists-in-python>

我们经常需要处理 csv 文件来分析与业务问题相关的数据。在本文中，我们将讨论如何将 csv 文件读入 python 中的[列表。](https://www.pythonforbeginners.com/basics/list-of-lists-in-python)

## 使用 CSV.reader()将 CSV 读入列表列表中

Python 为我们提供了 csv 模块来处理 python 中的 csv 文件。为了从一个`csv`文件中访问数据，我们经常使用一个借助于`csv.reader()`方法创建的 reader 对象。

创建 reader 对象后，我们可以将 csv 文件读入一个列表列表中。为此，我们将首先在读取模式下使用`open()`功能打开 csv 文件。`open()`函数将 csv 文件的文件名作为其第一个输入参数，将文字“`r`”作为其第二个输入参数，以表示该文件将以只读模式打开。执行后，`open()`方法返回一个引用 csv 文件的 file 对象。

现在，我们将 file 对象传递给`reader()`方法来创建一个 reader 对象。reader 对象实际上是一个迭代器，它将 csv 文件中的每一行都包含为一个列表。我们可以使用 for 循环访问 csv 文件的每一行，并将它读入如下的列表列表中。

```py
import csv

myFile = open('Demo.csv', 'r')
reader = csv.reader(myFile)
myList = []
for record in reader:
    myList.append(record)
print("The list of lists is:")
print(myList)
```

输出:

```py
The list of lists is:
[['Roll', 'Name', 'Language'], ['1', 'Aditya', 'Python'], ['2', 'Sam', ' Java'], ['3', ' Chris', ' C++']]
```

如果您想跳过 csv 文件的文件头，您可以使用 `next()` 功能。当在迭代器上执行时，`next()`函数从迭代器返回一个元素，并将迭代器移动到下一个元素。在 for 循环之外，可以使用一次`next()`函数将 csv 文件读入一个没有头文件的列表，如下所示。

```py
import csv

myFile = open('Demo.csv', 'r')
reader = csv.reader(myFile)
print("The header is:")
print(next(reader))
myList = []
for record in reader:
    myList.append(record)
print("The list of lists is:")
print(myList) 
```

输出:

```py
The header is:
['Roll', 'Name', 'Language']
The list of lists is:
[['1', 'Aditya', 'Python'], ['2', 'Sam', ' Java'], ['3', ' Chris', ' C++']]
```

您可以使用`list()` 构造函数，而不是使用 for 循环从 reader 对象读取 csv。这里，我们将把 reader 对象传递给`list()`构造函数，它将返回如下所示的列表列表。

```py
import csv

myFile = open('Demo.csv', 'r')
reader = csv.reader(myFile)
myList = list(reader)
print("The list of lists is:")
print(myList)
```

输出:

```py
The list of lists is:
[['Roll', 'Name', 'Language'], ['1', 'Aditya', 'Python'], ['2', 'Sam', ' Java'], ['3', ' Chris', ' C++']]
```

类似地，您可以使用`list()`构造函数以及`next()`方法和`csv.reader()`方法来读取一个没有头文件的 csv 文件，如下所示。

```py
import csv

myFile = open('Demo.csv', 'r')
reader = csv.reader(myFile)
print("The header is:")
print(next(reader))
myList = list(reader)
print("The list of lists is:")
print(myList) 
```

输出:

```py
The header is:
['Roll', 'Name', 'Language']
The list of lists is:
[['1', 'Aditya', 'Python'], ['2', 'Sam', ' Java'], ['3', ' Chris', ' C++']] 
```

## 结论

在本文中，我们讨论了用 python 将 csv 文件读入列表列表的不同方法。要了解更多关于列表的知识，你可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。

我希望你喜欢阅读这篇文章。

快乐学习！