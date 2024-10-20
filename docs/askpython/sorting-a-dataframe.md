# 在 Python 中对数据帧进行排序—循序渐进

> 原文：<https://www.askpython.com/python-modules/pandas/sorting-a-dataframe>

嘿，读者们！在本文中，我们将详细关注 Python 中对数据帧的**排序。所以，让我们开始吧！**

* * *

## 使用 sort_values()函数对数据帧进行排序

[Python Pandas 模块](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)为我们提供了各种处理大数据记录的函数。在根据数据帧处理数据记录时，我们经常会遇到需要对数据进行排序并表示输出的情况。

就在这时，Python*pandas . data frame . sort _ values()*函数出现了。

`sort_values() function`以定制的方式按**升序**或**降序**对数据进行排序。

现在，让我们在下一节集中讨论函数的结构。

* * *

## Python 中 sort_values()函数的语法

看看下面的语法！

```py
pandas.DataFrame.sort_values(by, axis=0, ascending=True, kind=’mergesort’)

```

*   **by** :表示要排序的列的列表。
*   **轴** : 0 表示按行排序，1 表示按列排序。
*   **升序**:如果为真，则按升序对数据帧进行排序。
*   **种类**:可以有三个值:`Quicksort`、`mergesort` 或`heapsort`。

现在，让我们在下一节中关注 sort_values()函数的实现。

* * *

## Python 中对数据帧进行排序的示例代码

在这个例子中，我们已经用`pandas.dataframe()`函数初步创建了一个数据帧。此外，我们已经使用 sort_values()函数按降序对“RATE”列进行了排序。

**举例:**

```py
import pandas as pd
data = pd.DataFrame([[3,0,1], [4,4,4], [1,7,7], [10,10,10]],
     index=['Python', 'Java', 'C','Kotlin'],
     columns=['RATE','EE','AA'])

sort = data.sort_values("RATE", axis = 0, ascending = False)

print("Data before sorting:\n")
print(data)

print("Data after sorting:\n")
print(sort)

```

**输出:**

```py
Data before sorting:

        RATE  EE  AA
Python     3   0   1
Java       4   4   4
C          1   7   7
Kotlin    10  10  10
Data after sorting:

        RATE  EE  AA
Kotlin    10  10  10
Java       4   4   4
Python     3   0   1
C          1   7   7

```

在下面的例子中，我们按两列对上述数据帧进行了排序，即“EE”和“AA ”,如下所示。

**举例:**

```py
import pandas as pd
data = pd.DataFrame([[3,0,1], [4,4,4], [1,7,7], [10,10,10]],
     index=['Python', 'Java', 'C','Kotlin'],
     columns=['RATE','EE','AA'])

sort = data.sort_values(["EE","AA"], axis = 0, ascending = True)

print("Data before sorting:\n")
print(data)

print("Data after sorting:\n")
print(sort)

```

**输出:**

如下图所示，数据框分别按照列“EE”和“AA”以升序排序。

```py
Data before sorting:

        RATE  EE  AA
Python     3   0   1
Java       4   4   4
C          1   7   7
Kotlin    10  10  10
Data after sorting:

        RATE  EE  AA
Python     3   0   1
Java       4   4   4
C          1   7   7
Kotlin    10  10  10

```

* * *

## 结论

到此，我们就结束了这个话题。我们已经了解了 sort_values()函数对数据帧进行排序的功能。

如果你遇到任何问题，欢迎在下面评论。更多与 Python 相关的此类帖子，敬请关注，继续学习！

* * *

## 参考

*   [Python sort_values()函数—文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html)