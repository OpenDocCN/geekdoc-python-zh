# 在 Python 中连接数据帧

> 原文：<https://www.pythonforbeginners.com/basics/concatenate-dataframes-in-python>

我们使用 python 中的数据帧来处理和分析 python 中的表格数据。在本文中，我们将讨论如何在 python 中连接两个或多个数据帧。

## Python 中如何连接数据帧？

要在 python 中连接两个或多个数据帧，我们可以使用 pandas 模块中定义的`concat()`方法。`concat()`方法将一列数据帧作为其输入参数，并垂直连接它们。

我们还可以使用`concat()`方法的 axis 参数水平连接 python 中的数据帧。axis 参数的默认值为 0，表示数据帧将垂直连接。如果要水平连接数据帧，可以将值 1 传递给轴参数。

执行后，`concat()`方法将返回结果数据帧。

## 在 python 中垂直连接数据帧

要在 python 中垂直连接两个数据帧，首先需要使用 import 语句导入 pandas 模块。之后，您可以使用`concat()` 方法连接数据帧，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade2.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2])
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Class  Roll    Name  Marks Grade
0      1    11  Aditya     85     A
1      1    12   Chris     95     A
2      1    14     Sam     75     B
3      1    16  Aditya     78     B
4      1    15   Harry     55     C
5      2     1    Joel     68     B
6      2    22     Tom     73     B
7      2    15    Golu     79     B
second dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
Merged dataframe is:
   Class  Roll        Name  Marks Grade
0      1    11      Aditya     85     A
1      1    12       Chris     95     A
2      1    14         Sam     75     B
3      1    16      Aditya     78     B
4      1    15       Harry     55     C
5      2     1        Joel     68     B
6      2    22         Tom     73     B
7      2    15        Golu     79     B
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
```

如果所有数据帧具有相同的列数，并且列名也相同，则结果数据帧具有与输入数据帧相同的列数。你可以在上面的例子中观察到这一点。

然而，如果一个数据帧的列数少于其他数据帧，则对于从该数据帧获得的行，该列在结果数据帧中的对应值将是 NaN。您可以在下面的示例中观察到这一点。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2])
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    20     72     B
7    24     92     A
second dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
Merged dataframe is:
   Roll  Marks Grade      Name
0    11     85     A       NaN
1    12     95     A       NaN
2    13     75     B       NaN
3    14     75     B       NaN
4    16     78     B       NaN
5    15     55     C       NaN
6    20     72     B       NaN
7    24     92     A       NaN
0    11     85     A    Aditya
1    12     95     A     Chris
2    13     75     B       Sam
3    14     75     B      Joel
4    16     78     B       Tom
5    15     55     C  Samantha
6    20     72     B      Tina
7    24     92     A       Amy
```

如果数据帧具有不同的列名，则在结果数据帧中，每个列名将被分配一个单独的列。此外，对于所获得的数据帧中没有指定列的行，该列的结果数据帧中的相应值将是 NaN。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 在 Python 中水平连接数据帧

为了水平连接数据帧，我们将使用 axis 参数，并在`concat()` 方法中给出值 1 作为它的输入。执行后，`concat()`方法将返回水平连接的 dataframe，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],axis=1)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    20     72     B
7    24     92     A
second dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
Merged dataframe is:
   Roll  Marks Grade  Roll      Name  Marks Grade
0    11     85     A    11    Aditya     85     A
1    12     95     A    12     Chris     95     A
2    13     75     B    13       Sam     75     B
3    14     75     B    14      Joel     75     B
4    16     78     B    16       Tom     78     B
5    15     55     C    15  Samantha     55     C
6    20     72     B    20      Tina     72     B
7    24     92     A    24       Amy     92     A
```

如果正在连接的数据帧具有相同数量的记录，则生成的数据帧将不具有任何 NaN 值，如上例所示。但是，如果一个数据帧的行数少于另一个数据帧，则结果数据帧将具有 NaN 值。当连接参数设置为“外部”时，会出现这种情况。

## 结论

在本文中，我们讨论了如何用 python 连接两个[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)。要连接两个以上的数据帧，只需将数据帧添加到数据帧列表中，该列表作为输入提供给`concat()`方法。

要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[字典理解的文章。你可能也会喜欢这篇关于用 python](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 理解[列表的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)