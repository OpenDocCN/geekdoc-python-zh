# 使用 Pandas melt()和 pivot()函数融化和取消融化数据

> 原文：<https://www.askpython.com/python-modules/pandas/melt-pivot-data>

读者朋友们，你们好！本文将重点介绍如何使用 melt()和 pivot()函数来融合和取消融合 Pandas 数据框中的数据值。

所以，让我们开始吧！🙂

* * *

## 熊猫的融化和不融化数据——简明概述

在深入研究融化和不融化数据的概念之前，我想把你的注意力吸引到这个诱人的词上来——冰淇淋🙂

是啊！冰淇淋…当你的盘子里有冰淇淋，当你正要吃第一口时，你接到一个电话。冰淇淋显然会融化，变成奶昔。

类似地，现在考虑在数据框中融合数据值的概念。数据值的融合用于将数据值从较宽的格式配置和改变为更窄和更长的格式。熔化的基本目的是创建特定格式的数据帧，其中一个或多个数据列充当数据属性的标识符。

在这个场景中，剩余的数据变量实际上被认为是数据值，并且只存在两列:变量和值。

另一方面，我们对数据变量执行去融合，以将值恢复到原始格式。

理解了数据的融化和不融化，现在让我们来理解熊猫的功能，它使我们能够实现同样的功能。

* * *

## 1.融化熊猫的数据变量

为了对数据变量进行熔化， [Python Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)为我们提供了 melt()函数。

**语法**:

```py
pandas.melt(frame, id_vars=None, value_vars=None,
 var_name=None, value_name='value')

```

*   框架:需要融化的实际数据框架。
*   id_vars:将作为标识符的列名。
*   value_vars:将被塑造成值的所有变量名(标识符变量除外)。
*   value_name:列**值**的名称，默认为*值*。

**举例**:

在本例中，我们使用 data frame()函数创建了一个包含变量的数据框:City、ID 和 Fav。

此外，我们现在将整个数据帧传递给 melt()函数，传递 ID 作为标识符变量，City 和 Fav 作为值变量。

```py
import pandas as pd

data = {"City": ["Pune", "Satara", "Solapur"], "ID": [1, 2, 3], "Fav": ["1", "3", "10"]}

dataf = pd.DataFrame(data)
print("Before melting..")
print(dataf)

melt_df = pd.melt(dataf, id_vars=["ID"], value_vars=["City", "Fav"])
print("After melting..")
print(melt_df)

```

**输出**:

因此，熔化后的数据只有三列:ID(标识符变量)、变量和值。通过这种方式，它可以将数据帧从宽格式转换为长格式。

```py
Before melting..
      City  ID Fav
0     Pune   1   1
1   Satara   2   3
2  Solapur   3  10
After melting..
   ID variable    value
0   1     City     Pune
1   2     City   Satara
2   3     City  Solapur
3   1      Fav        1
4   2      Fav        3
5   3      Fav       10

```

我们还可以跳过原始数据框中的列，同时将它传递给 melt()函数以排除某些列。

```py
import pandas as pd

data = {"City": ["Pune", "Satara", "Solapur"], "ID": [1, 2, 3], "Fav": ["1", "3", "10"]}

dataf = pd.DataFrame(data)
print("Before melting..")
print(dataf)

melt_df = pd.melt(dataf, id_vars=["City"], value_vars=["Fav"])
print("After melting..")
print(melt_df)

```

**输出—**

这里，因为我们已经排除了变量 **ID** ，所以在融合数据变量时不考虑它。

```py
Before melting..
      City  ID Fav
0     Pune   1   1
1   Satara   2   3
2  Solapur   3  10
After melting..
      City variable value
0     Pune      Fav     1
1   Satara      Fav     3
2  Solapur      Fav    10

```

* * *

## 2.使用 Pandas pivot()函数取消数据值的融合

融合了数据变量之后，现在是时候恢复数据框架的形状了。同样，Python 为我们提供了 pivot()函数。

**语法**:

```py
pandas.pivot(index, columns) 

```

*   索引:需要应用的标签，以使新数据框的索引就位。
*   列:需要应用的标签，以使新数据框的列就位。

**举例**:

1.  首先，我们创建了一个包含 ID、City 和 Fav 列的数据框。
2.  然后，我们应用 melt 并使用针对 ID 变量的 melt()函数延长数据帧作为标识符，用**表达式**作为变量名，用**值**作为代表非 pivoted 变量的列名。
3.  最后，我们使用 pivot()函数取消数据融合，提供 ID 作为新数据框的索引集。

```py
import pandas as pd

data = {"City": ["Pune", "Satara", "Solapur"], "ID": [1, 2, 3], "Fav": ["1", "3", "10"]}

dataf = pd.DataFrame(data)
print("Before melting..")
print(dataf)

melt_df = pd.melt(dataf, id_vars=["ID"], value_vars=["City","Fav"], var_name="Expression", value_name="Value")
print("After melting..")
print(melt_df)

unmelt = melt_df.pivot(index='ID', columns='Expression')
print("Post unmelting..")
print(unmelt)

```

**输出—**

```py
Before melting..
      City  ID Fav
0     Pune   1   1
1   Satara   2   3
2  Solapur   3  10
After melting..
   ID Expression    Value
0   1       City     Pune
1   2       City   Satara
2   3       City  Solapur
3   1        Fav        1
4   2        Fav        3
5   3        Fav       10
Post unmelting..
              Value    
Expression     City Fav
ID
1              Pune   1
2            Satara   3
3           Solapur  10

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。更多与 Python 编程相关的帖子，请继续关注我们。

在那之前，学习愉快！！🙂