# 用 Python 将列表追加到 CSV 文件中

> 原文：<https://www.pythonforbeginners.com/basics/append-list-to-csv-file-in-python>

列表是 python 中最常用的数据结构之一。在本文中，我们将讨论如何在 python 中向 CSV 文件追加列表。

## 使用 csv.writer()在 Python 中将列表追加到 CSV 文件中

csv 模块为我们提供了对 CSV 文件执行各种操作的不同方法。要在 python 中将一个列表附加到 csv 文件中，我们可以使用 c `sv.writerow()`方法和`csv.writer()`方法。为此，我们将使用以下步骤。

*   首先，我们将使用`open()`函数在追加模式下打开一个 csv 文件。`open()`函数将文件名作为第一个输入参数，将文字“a”作为第二个输入参数，以表明文件将以追加模式打开。它返回一个 file 对象，包含由`open()` 函数打开的 csv 文件。
*   打开文件后，我们将使用`csv.writer()`方法创建一个`csv.writer`对象。`csv.writer()`方法将 file 对象作为输入参数，并返回一个 writer 对象。一旦创建了 writer 对象，我们就可以使用`csv.writerow()`方法将列表附加到 csv 文件中。
*   在 writer 对象上调用`csv.writerow()`方法时，该方法将一个列表作为其输入参数，并将其附加到 writer 对象引用的 csv 文件中。我们将把这个列表作为输入参数传递给`writerow()`方法。

在执行`writerow()`方法后，列表将被附加到 CSV 文件中。为了保存数据，您应该使用`close()`方法关闭文件。否则，不会将任何更改保存到 csv 文件中。

使用 csv.writer()方法将列表追加到 csv 文件的源代码如下。

```py
import csv

myFile = open('Demo.csv', 'r+')
print("The content of the csv file before appending is:")
print(myFile.read())
myList = [4, 'Joel','Golang']
print("The list is:")
print(myList)
writer = csv.writer(myFile)
writer.writerow(myList)
myFile.close()
myFile = open('Demo.csv', 'r')
print("The content of the csv file after appending is:")
print(myFile.read()) 
```

输出:

```py
The content of the csv file before appending is:
Roll,Name,Language
1,Aditya,Python
2,Sam, Java
3, Chris, C++

The list is:
[4, 'Joel', 'Golang']
The content of the csv file after appending is:
Roll,Name,Language
1,Aditya,Python
2,Sam, Java
3, Chris, C++

4,Joel,Golang
```

## 结论

在本文中，我们讨论了在 python 中向 [csv 文件添加列表的方法。在这种方法中，列表将被附加到 csv 文件中，而不管它与 csv 中的列相比是否具有相同数量的元素。因此，建议确保每个列表与 csv 文件中的列相比具有相同数量的元素。此外，您应该确保列表中元素的顺序应该与 csv 文件中的列一致。否则，附加到 csv 文件的数据将变得不一致，并会导致错误。](https://www.pythonforbeginners.com/basics/append-dictionary-to-csv-file-in-python)

要了解更多关于 python 中的列表，你可以阅读这篇关于 python 中的列表理解的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。