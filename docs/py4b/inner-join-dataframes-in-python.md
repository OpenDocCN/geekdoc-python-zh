# Python 中的内部连接数据帧

> 原文：<https://www.pythonforbeginners.com/basics/inner-join-dataframes-in-python>

内部联接操作在数据库管理中用于联接两个或多个表。我们还可以对两个 [pandas 数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)执行内部连接操作，因为它们包含表格值。在本文中，我们将讨论如何在 python 中对两个数据帧执行内部连接操作。

## 什么是内部联接操作？

内部连接操作用于查找两个表之间的交集。例如，假设我们有一个包含学生个人详细信息的表和另一个包含学生成绩的表。如果两个表都有一个公共列，比如说`‘Name’`，那么我们可以创建另一个表，其中包含学生的详细信息以及他们在每一行的分数。

为了在 python 中执行内部连接操作，我们可以使用 pandas 数据帧以及`join()`方法或`merge()`方法。让我们逐一讨论。

程序中使用的文件可以通过下面的链接下载。

## 使用 merge()方法内部连接两个数据帧

我们可以使用`merge()`方法在 python 中对两个数据帧进行内连接操作。当在一个数据帧上调用时，`merge()`方法将另一个数据帧作为它的第一个输入参数。同时，它将值 `‘inner’`作为 `‘how’` 参数的输入参数。它还将两个数据帧之间的公共列名作为`‘on’`参数的输入变量。执行后，它返回一个数据帧，该数据帧是两个数据帧的交集，并且包含两个数据帧中的列。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
names=pd.read_csv("name.csv")
grades=pd.read_csv("grade.csv")
resultdf=names.merge(grades,how="inner",on="Name")
print("The resultant dataframe is:")
print(resultdf)
```

输出:

```py
The resultant dataframe is:
   Class_x  Roll_x    Name  Class_y  Roll_y Grade
0        1      11  Aditya        1      11     A
1        1      12   Chris        1      12    A+
2        2       1    Joel        2       1     B
3        2      22     Tom        2      22    B+
4        3      33    Tina        3      33    A-
5        3      34     Amy        3      34     A
```

您应该记住，输出数据帧将只包含两个表中的那些行，其中作为输入给 `‘on’`参数的列是相同的。来自两个数据帧的所有其他行将从输出数据帧中省略。

如果有同名的列，python 解释器会给列名添加 `_x` 和`_y`后缀。为了识别数据帧中调用了`merge()`方法的列，添加了`_x`后缀。对于作为输入参数传递给`merge()` 方法的 dataframe，使用了`_y`后缀。

**建议阅读:**如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章](https://codinginfinite.com/regression-in-machine-learning-with-examples/)。您可能也会喜欢这篇关于带有数字示例的 [k-means 聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 使用 Join()方法内部连接两个数据帧

不使用`merge()`方法，我们可以使用`join()`方法在数据帧上执行内部连接操作。

当在一个数据帧上调用时，`join()`方法将另一个数据帧作为它的第一个输入参数。同时，它将值`‘inner’` 作为 `‘how’` 参数的输入参数。它还将两个数据帧之间的公共列名作为`‘on’`参数的输入参数。执行后， `join()`方法返回如下所示的输出数据帧。

```py
import pandas as pd
import numpy as np
names=pd.read_csv("name.csv")
grades=pd.read_csv("grade.csv")
grades=grades.set_index("Name")
resultdf=names.join(grades,how="inner",on="Name",lsuffix='_names', rsuffix='_grades')
print("The resultant dataframe is:")
print(resultdf)
```

输出:

```py
The resultant dataframe is:
   Class_names  Roll_names    Name  Class_grades  Roll_grades Grade
0            1          11  Aditya             1           11     A
1            1          12   Chris             1           12    A+
3            2           1    Joel             2            1     B
4            2          22     Tom             2           22    B+
6            3          33    Tina             3           33    A-
7            3          34     Amy             3           34     A
```

在使用`join()`方法时，您还需要记住，要执行连接操作的列应该是作为输入参数传递给`join()`方法的 dataframe 的索引。如果数据帧的某些列有相同的列名，您需要使用`lsuffix`和`rsuffix`参数指定列名的后缀。如果列名相同，传递给这些参数的值可以帮助我们识别哪个列来自哪个数据帧。

## 结论

在本文中，我们讨论了在 python 中对两个数据帧执行内部连接操作的两种方法。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。