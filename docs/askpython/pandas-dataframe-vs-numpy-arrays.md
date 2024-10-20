# Pandas 数据帧和 Numpy 数组的区别

> 原文：<https://www.askpython.com/python/pandas-dataframe-vs-numpy-arrays>

我们经常混淆 Python 中的数据结构，因为它们看起来有些相似。Python 中的数据帧和数组是两种非常重要的数据结构，在数据分析中非常有用。在这篇文章中，我们将学习 Python 中 Pandas DataFrame 和 Numpy Array 的区别。

让我们从理解 Numpy 数组开始。

***也读作:[将熊猫 DataFrame 转换为 Numpy 数组【分步】](https://www.askpython.com/python-modules/numpy/pandas-dataframe-to-numpy-array)***

## 什么是 Numpy 数组？

NumPy 数组是 Python 中的一种多维数据结构，可以存储相似数据类型的对象。数组的元素由非负或正整数索引。数组是可变的，这意味着数组在形成后可以改变。数组在对向量进行数学运算时非常有用。它们为执行向量运算提供了许多有用的方法。

让我们看看如何创建一个数组。

我们将使用 Python 中的 Numpy 库。

```py
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)

```

**输出:**

```py
[1, 2, 3,4, 5]

```

现在让我们看看熊猫数据帧是什么。

## 什么是数据帧？

DataFrame 是 Python 中的一种二维表格可变数据结构，可以存储包含不同数据类型对象的表格数据。数据帧具有以行和列的形式标记的轴。数据帧是数据预处理的有用工具，因为它为数据处理提供了有用的方法。数据框架对于创建数据透视表和使用 Matplotlib 绘图也非常有用。

让我们看看如何在 Pandas 中创建数据帧。

```py
import pandas as pd
# Creating a dictionary
data = {'Name':["Tommy","Linda","Justin","Brendon"], 'Age':[31,24,16,22]}
df=pd.DataFrame(data)
print(df)

```

**输出:**

```py
      Name    Age
0    Tommy   31
1    Linda   24
2   Justin   16
3  Brendon   22

```

## 数据帧和数组的比较

下面列出了数据帧和数组之间的主要区别:

1.  [Numpy 数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)可以是多维的，而 DataFrame 只能是二维的。
2.  数组包含相似类型的对象或元素，而数据帧可以有对象或多种或相似的数据类型。
3.  数组和数据帧都是可变的。
4.  数组中的元素只能使用整数位置来访问，而数据帧中的元素可以使用整数和索引位置来访问。
5.  数据帧主要是以 SQL 表的形式出现，与表格数据相关联，而数组与数字数据和计算相关联。
6.  数据帧可以处理动态数据和混合数据类型，而数组不具备处理此类数据的灵活性。

## 结论

在本文中，您了解了 Pandas DataFrame 和 Numpy Array 之间的区别。Numpy 数组专门用于复杂的科学计算，而数据帧主要用于数据预处理。虽然这两种数据结构在数据分析中都起着非常重要的作用。