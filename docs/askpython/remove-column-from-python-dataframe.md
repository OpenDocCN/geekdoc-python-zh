# 从 Python 数据帧中删除列的 3 种简单方法

> 原文：<https://www.askpython.com/python-modules/pandas/remove-column-from-python-dataframe>

读者朋友们，你们好！在本文中，我们将关注从 Python 数据帧中移除列的**方法。那么，让我们开始吧。**

* * *

## 首先，什么是数据帧？

所以，伙计们！最终在走向解决方案之前，我们非常有必要理解和回忆一个数据框架的意义和存在。

一个[数据帧](https://www.askpython.com/python-modules/pandas/dataframes-in-python)是由 [Python 熊猫模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)提供的数据结构。它以行和列的形式存储值。因此，我们可以用矩阵的形式将数据表示为行和列。

数据帧类似于现实世界中的 Excel 或 CSV 文件。

* * *

## 如何从 Python 数据帧中删除列？

因此，了解了数据帧之后，现在让我们来关注一下从数据帧中完全删除列的技术。

### 1.Python dataframe.pop()方法

我们可以使用`pandas.dataframe.pop()`方法从数据框中移除或删除一列，只需提供列名作为参数。

**语法:**

```py
pandas.dataframe.pop('column-name')

```

**举例:**

```py
import pandas as pd 
data = {"Roll-num": [10,20,30,40,50,60,70], "Age":[12,14,13,12,14,13,15], "NAME":['John','Camili','Rheana','Joseph','Amanti','Alexa','Siri']}
block = pd.DataFrame(data)
print("Original Data frame:\n")
print(block)
block.pop('NAME')
print("\nData frame after deleting the column 'NAME':\n")
print(block)

```

这里，我们创建了一个 [Python 字典](https://www.askpython.com/python/dictionary/python-dictionary-dict-tutorial)作为‘数据’,并使用`pandas.DataFrame()`方法将其进一步转换成数据帧。

此外，我们应用了`pop()`方法来删除该列。

**输出:**

```py
Original Data frame:

   Roll-num  Age    NAME
0        10   12    John
1        20   14  Camili
2        30   13  Rheana
3        40   12  Joseph
4        50   14  Amanti
5        60   13   Alexa
6        70   15    Siri

Data frame after deleting the column 'NAME':

   Roll-num  Age
0        10   12
1        20   14
2        30   13
3        40   12
4        50   14
5        60   13
6        70   15

```

* * *

### 2.Python del 关键字删除该列

Python [del 关键字](https://www.askpython.com/python/dictionary/delete-a-dictionary-in-python)也可以用来直接从数据框中刷新列。`del keyword`通常用于删除或清除 Python 中的对象。

看看下面的语法！

**语法:**

```py
del dataframe['column-name']

```

**举例:**

```py
import pandas as pd 
data = {"Roll-num": [10,20,30,40,50,60,70], "Age":[12,14,13,12,14,13,15], "NAME":['John','Camili','Rheana','Joseph','Amanti','Alexa','Siri']}
block = pd.DataFrame(data)
print("Original Data frame:\n")
print(block)
del block["NAME"]
print("\nData frame after deleting the column 'NAME':\n")
print(block)

```

**输出:**

```py
Original Data frame:

   Roll-num  Age    NAME
0        10   12    John
1        20   14  Camili
2        30   13  Rheana
3        40   12  Joseph
4        50   14  Amanti
5        60   13   Alexa
6        70   15    Siri

Data frame after deleting the column 'NAME':

   Roll-num  Age
0        10   12
1        20   14
2        30   13
3        40   12
4        50   14
5        60   13
6        70   15

```

* * *

### 3.Python drop()函数删除列

`pandas.dataframe.drop() function`使我们能够从数据帧中删除值。这些值可以是面向行的，也可以是面向列的。

看看下面的语法！

```py
dataframe.drop('column-name', inplace=True, axis=1)

```

*   `inplace`:通过将它设置为**真**，改变被存储到一个新的被创建的对象中，并且它不改变原始的数据帧。
*   `axis` : **1** 用于列操作， **0** 用于行操作。

**举例:**

```py
import pandas as pd 
data = {"Roll-num": [10,20,30,40,50,60,70], "Age":[12,14,13,12,14,13,15], "NAME":['John','Camili','Rheana','Joseph','Amanti','Alexa','Siri']}
block = pd.DataFrame(data)
print("Original Data frame:\n")
print(block)
block.drop('NAME', inplace=True, axis=1)
print("\nData frame after deleting the column 'NAME':\n")
print(block)

```

**输出:**

```py
Original Data frame:

   Roll-num  Age    NAME
0        10   12    John
1        20   14  Camili
2        30   13  Rheana
3        40   12  Joseph
4        50   14  Amanti
5        60   13   Alexa
6        70   15    Siri

Data frame after deleting the column 'NAME':

   Roll-num  Age
0        10   12
1        20   14
2        30   13
3        40   12
4        50   14
5        60   13
6        70   15

```

* * *

## 结论

到此，我们就到了本文的结尾。希望这篇文章能更好地洞察你的兴趣。

如果你遇到任何问题，欢迎在下面评论。在那之前，学习愉快！！🙂

* * *

## 参考

*   [从 pandas 数据帧中删除列— StackOverFlow](https://stackoverflow.com/questions/13411544/delete-a-column-from-a-pandas-dataframe)