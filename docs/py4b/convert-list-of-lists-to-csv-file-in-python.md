# 用 Python 将列表转换成 CSV 文件

> 原文：<https://www.pythonforbeginners.com/lists/convert-list-of-lists-to-csv-file-in-python>

列表是 python 中最常用的数据结构之一。在本文中，我们将讨论如何用 python 将列表转换成 CSV 文件。

## 使用 csv.writer()将列表列表转换为 Python 中的 CSV

csv 模块为我们提供了对 CSV 文件执行各种操作的不同方法。要在 python 中将列表列表转换为 csv，我们可以使用 csv.writer()方法和 csv.writerow()方法。为此，我们将使用以下步骤。

*   首先，我们将使用 open()函数以写模式打开一个 csv 文件。open()函数将文件名作为第一个输入参数，将文字“w”作为第二个输入参数，以表明文件将以写模式打开。它返回一个 file 对象，该对象包含由 open()函数创建的空 csv 文件。
*   打开文件后，我们将使用 csv.writer()方法创建一个 csv.writer 对象。csv.writer()方法将 file 对象作为输入参数，并返回一个 writer 对象。一旦创建了 writer 对象，我们就可以使用 csv.writerow()方法将列表中的数据添加到 csv 文件中。
*   在 writer 对象上调用 csv.writerow()方法时，该方法获取一个值列表，并将其添加到 writer 对象引用的 csv 文件中。
*   首先，我们将为 CSV 文件添加文件头。为此，我们将把列名列表传递给 writerow()方法
*   添加头之后，我们将使用一个带有 writerow()方法的 for 循环将每个列表添加到 csv 文件中。这里，我们将把每个列表逐个传递给 writerow()方法。writerow()方法将列表添加到 csv 文件中。

执行 for 循环后，列表中的数据将被添加到 CSV 文件中。要保存数据，应该使用 close()方法关闭文件。否则，不会将任何更改保存到 csv 文件中。

使用 csv.writer()方法将列表转换为 csv 文件的源代码如下。

```py
import csv

listOfLists = [["Aditya", 1, "Python"], ["Sam", 2, 'Java'], ['Chris', 3, 'C++'], ['Joel', 4, 'TypeScript']]
print("THe list of lists is:")
print(listOfLists)
myFile = open('demo_file.csv', 'w')
writer = csv.writer(myFile)
writer.writerow(['Name', 'Roll', 'Language'])
for data_list in listOfLists:
    writer.writerow(data_list)
myFile.close()
myFile = open('demo_file.csv', 'r')
print("The content of the csv file is:")
print(myFile.read())
myFile.close()
```

输出:

```py
THe list of lists is:
[['Aditya', 1, 'Python'], ['Sam', 2, 'Java'], ['Chris', 3, 'C++'], ['Joel', 4, 'TypeScript']]
The content of the csv file is:
Name,Roll,Language
Aditya,1,Python
Sam,2,Java
Chris,3,C++
Joel,4,TypeScript 
```

## 结论

在本文中，我们讨论了一种用 python 将列表转换成 csv 文件的方法。在这些方法中，每个列表都将被添加到 csv 文件中，而不管它与 csv 中的列相比是否具有相同数量的元素。因此，建议确保每个元素都有相同数量的元素。此外，您应该确保列表中元素的顺序应该相同。否则，附加到 csv 文件的数据将变得不一致，并会导致错误。

要了解更多关于 python 中的列表，你可以阅读这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。