# Python 中的左连接数据帧

> 原文：<https://www.pythonforbeginners.com/basics/left-join-dataframes-in-python>

左连接操作在 SQL 中用于连接两个表。在本文中，我们将讨论如何在 python 中对两个数据帧执行左连接操作。

## 什么是左连接运算？

假设我们有两个表 A 和 B。当我们执行操作(左连接 B)时，我们得到一个新表，它包含表 A 中的所有行以及表 B 中的相应行。除此之外，表 A 中与表 B 中没有任何匹配行的所有行也包含在输出表中。但是，属于表 B 的行，在表 A 中没有任何匹配的行，将从最终结果中忽略。

为了理解这一点，假设我们有一个包含学生详细信息的行的表 A，和一个包含学生成绩的行的表 B。同样，两个表有一个公共列，比如说 `‘Name’`。现在，如果我们想执行操作 A left join B，结果表将包含学生的详细信息以及他们的分数。此外，这些学生的详细信息将在输出表中提及，其分数不在表 b 中。相反，这些学生的分数将不包括在输出表中，其详细信息不在表 a 中。

由于数据帧包含表格数据，我们可以在 python 中对数据帧执行左连接操作。为此，我们将使用`merge()`方法和`join()`方法。

您可以使用以下链接下载程序中使用的文件。

## 使用 merge()方法左连接数据帧

我们可以使用 python 中的`merge()`方法对数据帧执行左连接操作。为此，我们将在第一个数据帧上调用`merge()`方法。此外，我们将把第二个数据帧作为第一个输入参数传递给`merge()`方法。此外，我们将要匹配的列的名称作为输入参数传递给 `‘on’`参数，并将文字`‘left’`作为输入参数传递给`‘how’`参数。执行后， `merge()`方法将返回输出数据帧，如下例所示。

```py
import pandas as pd
import numpy as np
names=pd.read_csv("name.csv")
grades=pd.read_csv("grade.csv")
resultdf=names.merge(grades,how="left",on="Name")
print("The resultant dataframe is:")
print(resultdf)
```

输出:

```py
The resultant dataframe is:
   Class_x  Roll_x      Name  Class_y  Roll_y Grade
0        1      11    Aditya      1.0    11.0     A
1        1      12     Chris      1.0    12.0    A+
2        1      13       Sam      NaN     NaN   NaN
3        2       1      Joel      2.0     1.0     B
4        2      22       Tom      2.0    22.0    B+
5        2      44  Samantha      NaN     NaN   NaN
6        3      33      Tina      3.0    33.0    A-
7        3      34       Amy      3.0    34.0     A 
```

如果第一个数据帧中的行在第二个数据帧中没有匹配的数据帧，这些行仍会包含在输出中。然而，对于在第一数据帧中没有任何匹配行的第二数据帧中的行来说，情况并非如此。你可以在上面的例子中观察到这一点。

如果有同名的列，python 解释器会给列名添加 `_x` 和`_y`后缀。为了识别数据帧中调用了`merge()`方法的列，添加了`_x`后缀。对于作为输入参数传递给`merge()` 方法的 dataframe，使用了`_y`后缀。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 使用 Join()方法左连接数据帧

不使用`merge()` 方法，我们可以使用 `join()`方法在给定的数据帧上执行左连接操作。当在一个数据帧上调用 `join()`方法时，它将另一个数据帧作为它的第一个输入参数。此外，我们将要匹配的列的名称作为输入参数传递给`‘on’`参数，并将文字`‘left’`作为输入参数传递给`‘how’`参数。执行后， `join()`方法返回输出数据帧，如下例所示。

```py
import pandas as pd
import numpy as np
names=pd.read_csv("name.csv")
grades=pd.read_csv("grade.csv")
grades=grades.set_index("Name")
resultdf=names.join(grades,how="left",on="Name",lsuffix='_names', rsuffix='_grades')
print("The resultant dataframe is:")
print(resultdf)
```

输出:

```py
The resultant dataframe is:
   Class_names  Roll_names      Name  Class_grades  Roll_grades Grade
0            1          11    Aditya           1.0         11.0     A
1            1          12     Chris           1.0         12.0    A+
2            1          13       Sam           NaN          NaN   NaN
3            2           1      Joel           2.0          1.0     B
4            2          22       Tom           2.0         22.0    B+
5            2          44  Samantha           NaN          NaN   NaN
6            3          33      Tina           3.0         33.0    A-
7            3          34       Amy           3.0         34.0     A
```

在使用`join()`方法时，您还需要记住，要执行连接操作的列应该是作为输入参数传递给`join()`方法的 dataframe 的索引。如果数据帧的某些列有相同的列名，您需要使用`lsuffix`和`rsuffix`参数指定列名的后缀。如果列名相同，传递给这些参数的值可以帮助我们识别哪个列来自哪个数据帧。

## 结论

在本文中，我们讨论了在 python 中对数据帧执行左连接操作的两种方法。想了解更多关于 python 编程的知识，可以阅读这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。