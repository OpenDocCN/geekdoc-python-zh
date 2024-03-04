# 用 Python 将字典附加到 CSV 文件

> 原文：<https://www.pythonforbeginners.com/basics/append-dictionary-to-csv-file-in-python>

CSV 文件是存储结构化表格数据的最有效工具之一。有时，我们可能需要从 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中向 CSV 文件追加数据。在本文中，我们将讨论如何用 python 将字典中的值追加到 CSV 文件中。

## 使用 csv.writer()将字典附加到 CSV 文件

您可以使用`CSV.writer()`方法和`CSV.writerow()` 方法将字典附加到 CSV 文件。为此，我们将首先使用`open()` 函数在追加模式下打开 CSV 文件。`open()`函数将文件名作为其第一个输入参数，将文字“`a`”作为其第二个输入参数，以表示该文件是以追加模式打开的。

打开文件后，我们将使用`CSV.writer()`函数创建一个 CSV writer 对象。`CSV.writer()`函数将包含 CSV 文件的 file 对象作为其输入参数，并返回一个 writer 对象。

创建 writer 对象后，我们将使用`w` riterow()方法将字典附加到 CSV 文件中。在 writer 对象上调用`writerow()`方法时，该方法将字典中的值作为其输入参数，并将其附加到 CSV 文件中。

在执行了`writerow()`方法之后，您必须使用`close()` 方法关闭 CSV 文件。否则，更改不会保存在 CSV 文件中。下面给出了这种向 CSV 文件添加字典的方法的源代码。

```py
import csv

myFile = open('Demo.csv', 'r+')
print("The content of the csv file before appending is:")
print(myFile.read())
myDict = {'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
print("The dictionary is:")
print(myDict)
writer = csv.writer(myFile)
writer.writerow(myDict.values())
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

The dictionary is:
{'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
The content of the csv file after appending is:
Roll,Name,Language
1,Aditya,Python
2,Sam, Java
3, Chris, C++
4,Joel,Golang
```

## 使用 csv 将字典附加到 CSV 文件。词典作者()

不使用`csv.writer()`方法，我们可以使用`csv.DictWriter()`函数和`csv.writerow()` 方法将 python 字典附加到 csv 文件中。该方法几乎类似于使用`csv.writer()`方法的方法，但有以下区别。

*   我们将使用`csv.DictWriter()`方法，而不是 `csv.writer()` 方法。`DictWriter()`方法将包含 csv 文件的 file 对象作为其输入参数，并返回一个 DictWriter 对象。
*   当在 DictWriter 对象上执行`writerow()`方法时，它将一个字典作为输入参数，而不是字典中的值。

使用`csv.DictWriter()` 将字典附加到 csv 文件的 python 代码如下。

```py
import csv

myFile = open('Demo.csv', 'r+')
print("The content of the csv file before appending is:")
print(myFile.read())
myDict = {'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
print("The dictionary is:")
print(myDict)
writer = csv.DictWriter(myFile, fieldnames=list(myDict.keys()))
writer.writerow(myDict)
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

The dictionary is:
{'Roll': 4, 'Name': 'Joel', 'Language': 'Golang'}
The content of the csv file after appending is:
Roll,Name,Language
1,Aditya,Python
2,Sam, Java
3, Chris, C++

4,Joel,Golang
```

## 结论

在本文中，我们讨论了用 python 向 csv 文件添加字典的两种方法。在这些方法中，将追加字典，而不管它与 csv 文件中的列相比是否具有相同的项目数，或者它与 csv 文件中的列名相比是否具有相同的键。因此，它建议确保字典应该具有与 csv 文件中的列相同数量的键。您还应该确保 csv 文件的列顺序应该与字典中的键顺序相同。否则，附加到 csv 文件的数据将变得不一致，并会导致错误。

我希望你喜欢阅读这篇文章。想了解更多 python 中的字典，可以阅读这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你也可以用 python 写这篇关于[列表理解的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)

请继续关注更多内容丰富的文章。

快乐学习！