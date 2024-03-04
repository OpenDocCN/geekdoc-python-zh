# 用 Python 在数据帧中追加新行

> 原文：<https://www.pythonforbeginners.com/basics/append-a-new-row-in-a-dataframe-in-python>

为了在 python 中处理表格数据，我们通常使用 dataframes。在本文中，我们将讨论如何在数据帧中追加新行。

## 使用 loc[]属性在数据帧中追加新行

如果我们有一个列表形式的行，我们可以使用 pandas 模块中定义的 loc[]属性将该行添加到 dataframe 中。loc 属性用于获取数据帧中特定位置的行。当在数据帧上调用时，它将行号作为方括号内的输入，并返回数据帧的一部分。

为了将列表作为新行添加到数据帧中，我们将把列表分配给数据帧最后一行的片。

要获得数据帧中最后一行的位置，我们可以使用 len()函数来确定数据帧的长度。获得数据帧的长度后，我们可以将列表作为新行添加到数据帧中，如下所示。

```py
import pandas as pd

df = pd.read_csv('Demo.csv')
print("The dataframe before the append operation:")
print(df)
values = [10, "Sam", "Typescript"]
length = len(df)
df.loc[length] = values
print("The dataframe after the append operation:")
print(df)
```

输出:

```py
The dataframe before the append operation:
   Roll    Name Language
0     1  Aditya   Python
1     2     Sam     Java
2     3   Chris      C++
The dataframe after the append operation:
   Roll    Name    Language
0     1  Aditya      Python
1     2     Sam        Java
2     3   Chris         C++
3    10     Sam  Typescript
```

在这种方法中，我们将一个列表作为新行添加到数据帧中。现在，让我们讨论一种将 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)附加到数据帧的方法。

## 使用 Append()方法在数据帧中追加新行

如果给我们一个字典，其中字典的键由数据帧的列名组成，我们可以使用`append()`方法将字典作为一行添加到数据帧中。在 dataframe 上调用 append()方法时，该方法将 python 字典作为其输入参数，并将字典的值追加到 dataframe 的最后一行。另外，我们需要将值 True 赋给 dataframe 的 ignore_index 参数。执行后，append()方法返回更新后的数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd

df = pd.read_csv('Demo.csv')
print("The dataframe before the append operation:")
print(df)
valueDict = {'Roll': 15, 'Name': "Wilson", 'Language': "Golang"}
length = len(df)
df = df.append(valueDict, ignore_index=True)
print("The dataframe after the append operation:")
print(df)
```

输出:

```py
The dataframe before the append operation:
   Roll    Name Language
0     1  Aditya   Python
1     2     Sam     Java
2     3   Chris      C++
The dataframe after the append operation:
   Roll    Name Language
0     1  Aditya   Python
1     2     Sam     Java
2     3   Chris      C++
3    15  Wilson   Golang
```

## 结论

在本文中，我们讨论了用 python 在数据帧中追加新行的两种方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[列表理解的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python) 理解[字典的文章。](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)