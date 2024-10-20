# 如何在 Python 中从一个数据帧中获取唯一值？

> 原文：<https://www.askpython.com/python/built-in-methods/unique-values-from-a-dataframe>

读者朋友们，你们好！在本文中，我们将关注如何在 Python 中从[数据帧中获取唯一值。](https://www.askpython.com/python-modules/pandas/dataframes-in-python)

所以，让我们开始吧！

* * *

## 什么是 Python 数据帧？

Python Pandas 模块为我们提供了各种数据结构和函数来存储和操作大量数据。

**DataFrame** 是 Pandas 模块提供的一个数据结构化工具，用于处理多维度的大型数据集，如巨大的 csv 或 excel 文件等。

由于我们可以在一个数据框中存储大量数据，因此我们经常会遇到这样的情况:从可能包含冗余或重复值的数据集中查找唯一的数据值。

这就是`pandas.dataframe.unique() function`出现的时候。

现在让我们在下一节集中讨论 unique()函数的功能。

* * *

## Python pandas.unique()函数从数据帧中获取唯一值

`pandas.unique() function`返回数据集中存在的唯一值。

**它基本上使用一种基于哈希表的技术，从数据帧/系列数据结构中的值集中返回非冗余值。**

让我们通过一个例子来理解独特功能的作用

考虑包含如下值的数据集:1，2，3，2，4，3，2

现在，如果我们应用 unique()函数，我们将获得以下结果:1，2，3，4。这样，我们可以很容易地找到数据集的唯一值。

现在，让我们在下一节讨论 pandas.unique()函数的结构。

* * *

### Python unique()函数的语法

看看下面的语法:

```py
pandas.unique(data)

```

当数据是一维的时，上述语法是有用的。它代表一维数据值中的唯一值(序列数据结构)。

但是，如果数据包含不止一个维度，即行和列，该怎么办呢？是的，我们在下面的语法中有一个解决方案

```py
pandas.dataframe.column-name.unique()

```

这种语法使我们能够从数据集的特定列中找到唯一的值。

数据最好是分类类型，这样唯一函数才能获得正确的结果。此外，数据按照其在数据集中出现的顺序显示。

* * *

## 熊猫系列的 Python unique()函数

在下面的例子中，我们创建了一个包含冗余值的列表。

此外，我们已经将该列表转换为一个系列数据结构，因为它只有一个维度。最后，我们应用了 unique()函数从数据中获取唯一值。

**举例:**

```py
lst = [1,2,3,4,2,4]
df = pandas.Series(lst)
print("Unique values:\n")
print(pandas.unique(df))

```

**输出:**

```py
Unique values:
[1 2 3 4]

```

* * *

## Python unique()函数与熊猫数据帧

让我们首先将数据集加载到如下所示的环境中

```py
import pandas
BIKE = pandas.read_csv("Bike.csv")

```

在 这里可以找到数据集 [**。**](https://github.com/Safa1615/BIKE-RENTAL-COUNT/blob/master/day.csv)

`pandas.dataframe.nunique() function`代表数据帧每一列中的唯一值。

```py
BIKE.nunique()

```

**输出:**

```py
season          4
yr              2
mnth           12
holiday         2
weathersit      3
temp          494
hum           586
windspeed     636
cnt           684
dtype: int64

```

此外，我们使用以下代码表示了“季节”列中呈现的独特值

```py
BIKE.season.unique()

```

**输出:**

```py
array([1, 2, 3, 4], dtype=int64)

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂