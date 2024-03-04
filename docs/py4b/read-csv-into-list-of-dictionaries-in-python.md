# 用 Python 将 CSV 读入字典列表

> 原文：<https://www.pythonforbeginners.com/basics/read-csv-into-list-of-dictionaries-in-python>

csv 文件用于存储结构化数据，CSV 文件中的每一行都存储一个条目，其中每个值都与列名相关联。类似地， [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)用于存储 python 中的键值对。在本文中，我们将讨论如何将 csv 文件读入 python 中的字典列表。

## 使用 csv 将 CSV 读入字典列表。字典阅读器()

在 python 中，我们可以使用 csv 模块来处理 csv 文件。为了将 csv 文件读入字典列表，我们将使用`csv.DictReader()`方法创建一个`csv.DictReader`对象。在创建了`DictReader`对象之后，我们可以使用以下步骤从 csv 文件创建一个字典列表。

*   首先，我们将在读取模式下使用`open()`函数打开 csv 文件。`open()` 函数将文件名作为它的第一个输入参数，将文字“r”作为它的第二个输入参数，以表明文件是以读取模式打开的。执行后，它返回一个包含 csv 文件的 file 对象。
*   从`open()` 函数获得文件对象后，我们将使用`csv.DictReader()`函数创建一个`DictReader`对象。`csv.DictReader()`函数将文件对象作为其输入参数，并返回一个`DictReader`对象。
*   `DictReader`对象作为迭代器工作，包含 csv 文件的每一行作为字典。在字典中，键由 csv 文件的列名组成，而与键相关联的值是行中特定列中的值。
*   要将 csv 文件读入字典列表，我们将首先创建一个空列表。之后，我们将使用 for 循环将每个字典从`DictReader`对象添加到列表中。在执行 for 循环之后，我们将获得整个 csv 文件作为字典列表。
*   不要忘记在程序结束时使用`close()`方法关闭文件。

使用`DictReader()`方法和 for 循环将 csv 读入字典列表的程序如下。

```py
import csv

myFile = open('Demo.csv', 'r')
reader = csv.DictReader(myFile)
myList = list()
for dictionary in reader:
    myList.append(dictionary)
print("The list of dictionaries is:")
print(myList)
```

输出:

```py
The list of dictionaries is:
[{'Roll': '1', 'Name': 'Aditya', 'Language': 'Python'}, {'Roll': '2', 'Name': 'Sam', 'Language': ' Java'}, {'Roll': '3', 'Name': ' Chris', 'Language': ' C++'}] 
```

除了使用 for 循环，还可以使用 `list()`构造函数将`DictReader`对象转换成一个列表，如下所示。

```py
import csv

myFile = open('Demo.csv', 'r')
reader = csv.DictReader(myFile)
myList = list(reader)
print("The list of dictionaries is:")
print(myList) 
```

输出:

```py
The list of dictionaries is:
[{'Roll': '1', 'Name': 'Aditya', 'Language': 'Python'}, {'Roll': '2', 'Name': 'Sam', 'Language': ' Java'}, {'Roll': '3', 'Name': ' Chris', 'Language': ' C++'}]
```

## 结论

在本文中，我们讨论了如何将 csv 文件读入 python 中的字典列表。要了解更多关于列表的知识，你可以阅读这篇关于 python 中的列表理解的文章。你可能也会喜欢这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。