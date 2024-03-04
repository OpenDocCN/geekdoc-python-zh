# Python 中数据帧的字典列表

> 原文：<https://www.pythonforbeginners.com/basics/list-of-dictionaries-to-dataframe-in-python>

在 python 中，数据帧主要用于分析表格数据。在本文中，我们将讨论如何用 python 将字典列表转换成数据帧。

## 使用熊猫数据框的字典列表。`DataFrame`()

dataframe 对象在 pandas 模块中定义。为了从给定的字典列表中创建一个数据帧，我们可以使用 `DataFrame()`方法。`DataFrame()`方法对象将字典列表作为输入参数，并返回从字典创建的 dataframe。这里，数据帧的列名由 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)中的键组成。每个字典的值被转换成数据帧的行。您可以使用 `pandas.DataFrame()` 方法从字典列表中创建一个数据帧，如下所示。

```py
import pandas as pd

listOfDict = [{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'},
              {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
print("THe list of dictionaries is:")
print(listOfDict)
df = pd.DataFrame(listOfDict)
print("The dataframe is:")
print(df)
```

输出:

```py
THe list of dictionaries is:
[{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'}, {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
```

在数据帧中，为字典列表中的每个关键字创建一列。如果一个字典没有任何特定的键，则值`NaN`被赋予对应于代表该字典的行中的特定键的列。

## 使用 pandas.dataframe.from_dict()方法将字典列表转换为 Python 中的数据帧

除了使用`DataFrame()` 方法，我们还可以使用`from_dict()` 方法将字典列表转换成 Python 中的数据帧。`from_dict()`方法将一个字典列表作为其输入参数，并返回一个 dataframe。同样，数据帧的列名由字典中的键组成。每个字典的值被转换成数据帧的行。您可以使用`from_dict()`方法从字典列表中创建一个数据帧，如下所示。

```py
import pandas as pd

listOfDict = [{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'},
              {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
print("THe list of dictionaries is:")
print(listOfDict)
df = pd.DataFrame.from_dict(listOfDict)
print("The dataframe is:")
print(df)
```

输出:

```py
THe list of dictionaries is:
[{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'}, {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
```

您可以观察到由 from_dict()方法生成的 dataframe 类似于由 dataframe()方法生成的 DataFrame。

## 使用 pandas . Dataframe . from _ records(Data)将字典列表转换为 Data frame

为了从 Python 中的字典列表创建一个 dataframe，我们也可以使用`from_records()`方法。`from_records()` 方法将字典列表作为其输入参数，并返回一个 dataframe。同样，数据帧的列名由字典中的键组成。每个字典的值被转换成数据帧的行。您可以使用`from_records()` 方法从字典列表中创建一个数据帧，如下所示。

```py
import pandas as pd

listOfDict = [{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'},
              {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
print("THe list of dictionaries is:")
print(listOfDict)
df = pd.DataFrame.from_records(listOfDict)
print("The dataframe is:")
print(df)
```

输出:

```py
THe list of dictionaries is:
[{'Name': 'Aditya', 'Roll': 1, 'Language': 'Python'}, {'Name': 'Sam', 'Roll': 2, 'Language': 'Java'}, {'Name': 'Chris', 'Roll': 3, 'Language': 'C++'}, {'Name': 'Joel', 'Roll': 4, 'Language': 'TypeScript'}]
The dataframe is:
     Name  Roll    Language
0  Aditya     1      Python
1     Sam     2        Java
2   Chris     3         C++
3    Joel     4  TypeScript
```

## 结论

在本文中，我们讨论了用 python 将字典列表转换成数据帧的三种方法。这三种方法在语义上是相似的，并且产生相同的结果。要了解更多关于字典的知识，你可以阅读这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。