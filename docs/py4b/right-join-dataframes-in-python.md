# Python 中的右连接数据帧

> 原文：<https://www.pythonforbeginners.com/basics/right-join-dataframes-in-python>

右连接操作用于连接 SQL 中的两个表。在本文中，我们将讨论如何在 python 中对两个数据帧执行正确的连接操作。

## 什么是正确的连接操作？

考虑两个表 A 和 B，其中 A 包含一个班级中学生的详细信息，而表 B 包含学生的分数。表 A 和表 B 都有一个公共列“Name”。当我们对表执行(A right join B)操作时，我们得到一个包含表 B 中所有行以及表 A 中相应行的表。除此之外，表 B 中与表 A 中没有任何匹配行的所有行也包含在输出表中。但是，属于表 A 的行，在表 B 中没有任何匹配的行，将从最终结果中忽略。

因此，我们将获得一个新表，其中包含个人详细信息以及表 b 中给出分数的学生的分数。输出表还将包含表 a 中未给出详细信息的学生的分数。但是，输出将不包含表 b 中未给出分数的学生的详细信息。

我们还可以对 pandas 数据帧执行右连接操作，因为数据帧包含表格形式的数据。为此，我们可以使用本文中讨论的`merge()`方法和`join()`方法。

您可以使用以下链接下载程序中使用的文件。

## 使用 Python 中的 merge()方法右连接数据帧

我们可以使用 python 中的`merge()`方法对数据帧执行正确的连接操作。为此，我们将在第一个数据帧上调用`merge()`方法。此外，我们将把第二个数据帧作为第一个输入参数传递给`merge()`方法。此外，我们将要匹配的列的名称作为输入参数传递给`‘on’`参数，并将文字`‘right’`作为输入参数传递给`‘how’`参数。执行后，`merge()` 方法将返回输出数据帧，如下例所示。

```py
import pandas as pd
import numpy as np
names=pd.read_csv("name.csv")
grades=pd.read_csv("grade.csv")
resultdf=names.merge(grades,how="right",on="Name")
print("The resultant dataframe is:")
print(resultdf)
```

输出:

```py
The resultant dataframe is:
   Class_x  Roll_x        Name  Class_y  Roll_y Grade
0      1.0    11.0      Aditya        1      11     A
1      1.0    12.0       Chris        1      12    A+
2      2.0     1.0        Joel        2       1     B
3      2.0    22.0         Tom        2      22    B+
4      3.0    33.0        Tina        3      33    A-
5      3.0    34.0         Amy        3      34     A
6      NaN     NaN  Radheshyam        3      23    B+
7      NaN     NaN       Bobby        3      11     D
```

如果第一个数据帧中的行在第二个数据帧中没有匹配的数据帧，则这些行不会包含在输出中。然而，对于在第一数据帧中没有任何匹配行的第二数据帧中的行来说，情况并非如此。第二个数据帧的所有行都将包括在输出中，即使它们在第一个数据帧中没有任何匹配的行。您可以在下面的示例中观察到这一点。

如果两个数据帧中有同名的列，python 解释器会在列名中添加 `_x` 和`_y`后缀。为了识别数据帧中调用了`merge()`方法的列，添加了`_x`后缀。对于作为输入参数传递给`merge()` 方法的 dataframe，使用了`_y`后缀。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 使用 Python 中的 Join()方法右连接数据帧

不使用`merge()`方法，我们可以使用`join()` 方法在给定的数据帧上执行正确的连接操作。当在一个数据帧上调用 `join()` 方法时，它将另一个数据帧作为它的第一个输入参数。此外，我们将把要匹配的列的名称作为输入参数传递给`‘on’`参数，把文字`“right”`作为输入参数传递给'`how`'参数。执行后，`join()` 方法返回输出数据帧，如下例所示。

```py
import pandas as pd
import numpy as np
names=pd.read_csv("name.csv")
grades=pd.read_csv("grade.csv")
grades=grades.set_index("Name")
resultdf=names.join(grades,how="right",on="Name",lsuffix='_names', rsuffix='_grades')
print("The resultant dataframe is:")
print(resultdf)
```

输出:

```py
The resultant dataframe is:
     Class_names  Roll_names        Name  Class_grades  Roll_grades Grade
0.0          1.0        11.0      Aditya             1           11     A
1.0          1.0        12.0       Chris             1           12    A+
3.0          2.0         1.0        Joel             2            1     B
4.0          2.0        22.0         Tom             2           22    B+
6.0          3.0        33.0        Tina             3           33    A-
7.0          3.0        34.0         Amy             3           34     A
NaN          NaN         NaN  Radheshyam             3           23    B+
NaN          NaN         NaN       Bobby             3           11     D
```

在使用`join()`方法时，您需要记住，要执行连接操作的列应该是作为输入参数传递给`join()`方法的 dataframe 的索引。如果数据帧的某些列有相同的列名，您需要使用`lsuffix`和`rsuffix`参数指定列名的后缀。如果列名相同，传递给这些参数的值可以帮助我们识别哪个列来自哪个数据帧。

## 结论

在本文中，我们讨论了在 python 中对数据帧执行正确连接操作的两种方法。想了解更多关于 python 编程的知识，可以阅读这篇关于[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。