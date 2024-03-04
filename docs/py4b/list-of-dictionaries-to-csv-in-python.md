# Python 中 CSV 格式的词典列表

> 原文：<https://www.pythonforbeginners.com/basics/list-of-dictionaries-to-csv-in-python>

我们在 python 中使用字典来存储键值对。类似地，我们使用 CSV 文件来存储包含特定字段值的记录。在本文中，我们将讨论如何用 python 将词典列表转换成 CSV 文件。

## 使用 csv.writer()在 Python 中将字典列表转换为 CSV

csv 模块为我们提供了对 CSV 文件执行各种操作的不同方法。要在 python 中将字典列表转换成 csv 格式，我们可以使用`csv.writer()`方法和`csv.writerow()`方法。为此，我们将使用以下步骤。

*   首先，我们将使用`open()`函数以写模式打开一个 csv 文件。`open()`函数将文件名作为第一个输入参数，将文字“w”作为第二个输入参数，以表明文件将以写模式打开。它返回一个 file 对象，包含由`open()`函数创建的空 csv 文件。
*   打开文件后，我们将使用`csv.writer()`方法创建一个`csv.writer`对象。`csv.writer()`方法将 file 对象作为输入参数，并返回一个 writer 对象。一旦创建了 writer 对象，我们就可以使用`csv.writerow()`方法将字典列表中的数据添加到 csv 文件中。
*   在 writer 对象上调用`csv.writerow()`方法时，该方法获取一个值列表，并将其添加到 writer 对象引用的 csv 文件中。
*   首先，我们将通过向 csv 文件添加字典的键来添加 CSV 文件的头。
*   添加头之后，我们将使用一个带有`writerow()` 方法的 for 循环来将每个字典添加到 csv 文件中。这里，我们将把字典中的值传递给 CSV 文件。

在执行 for 循环后，来自 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)的数据将被添加到 CSV 文件中。要保存数据，您应该使用`close()`方法关闭文件。否则，不会将任何更改保存到 csv 文件中。

使用`csv.writer()`方法将字典列表转换成 csv 文件的源代码如下。

```py
import csv

listOfDict = [{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'},
              {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
print("THe list of dictionaries is:")
print(listOfDict)
myFile = open('demo_file.csv', 'w')
writer = csv.writer(myFile)
writer.writerow(['Name', 'Roll', 'Language'])
for dictionary in listOfDict:
    writer.writerow(dictionary.values())
myFile.close()
myFile = open('demo_file.csv', 'r')
print("The content of the csv file is:")
print(myFile.read())
myFile.close()
```

输出:

```py
THe list of dictionaries is:
[{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'}, {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
The content of the csv file is:
Name,Roll,Language
Aditya,1,Python
Sam,2,Java
Chris,3,C++
Joel,4,TypeScript
```

## 使用 csv 将字典列表转换为 Python 中的 CSV。词典作者()

我们可以一次将整个词典列表转换为 csv 文件，而不是使用迭代方法将每个词典添加到 CSV 文件中。为此，我们将使用`DictWriter()` 方法和`csv.writerows()`方法。

这种方法在以下几个步骤上不同于前面的方法。

*   我们将使用`Dictwriter()`方法创建一个`csv.DictWriter` 对象，而不是创建一个 csv.writer 对象。`DictWriter()`方法将包含 csv 文件的 file 对象作为第一个参数，将 csv 文件的列名作为第二个输入参数。执行后，它返回一个`DictWriter`对象。
*   创建完 DictWriter 对象后，我们将把头文件添加到 csv 文件中。为此，我们将使用`writeheader()`方法。当在一个`DictWriter`对象上执行时，`writeheader()`方法添加提供给`DictWriter()`方法的列作为 csv 文件的标题。
*   添加头之后，我们可以使用`writerows()`方法将整个词典列表添加到 csv 文件中。当在一个`DictWriter`对象上调用时，`writerows()`方法将一个字典列表作为其输入参数，并将字典中的值添加到 csv 文件中。

将整个词典列表添加到 csv 文件后，必须使用`close()`方法关闭该文件。否则，不会保存任何更改。

下面给出了用 python 将字典列表转换成 csv 文件的源代码。

```py
import csv

listOfDict = [{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'},
              {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
print("THe list of dictionaries is:")
print(listOfDict)
myFile = open('demo_file.csv', 'w')
writer = csv.DictWriter(myFile, fieldnames=['Name', 'Roll', 'Language'])
writer.writeheader()
writer.writerows(listOfDict)
myFile.close()
myFile = open('demo_file.csv', 'r')
print("The content of the csv file is:")
print(myFile.read())
myFile.close()
```

输出:

```py
THe list of dictionaries is:
[{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'}, {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
The content of the csv file is:
Name,Roll,Language
Aditya,1,Python
Sam,2,Java
Chris,3,C++
Joel,4,TypeScript
```

## 结论

在本文中，我们讨论了用 python 将词典列表转换成 csv 文件的两种方法。在这些方法中，每个字典都将被添加到 csv 文件中，而不管它与 csv 文件中的列相比是否具有相同的项目数，或者与 csv 文件中的列名相比是否具有相同的键。因此，建议确保每本词典都有相同数量的条目。此外，您应该确保字典中的键的顺序应该相同。否则，附加到 csv 文件的数据将变得不一致，并会导致错误。

想了解更多 python 中的字典，可以阅读这篇关于 python 中[字典理解的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)