# 在 Python 中合并数据帧

> 原文：<https://www.pythonforbeginners.com/basics/merge-dataframes-in-python>

Python 为我们提供了 [pandas dataframes](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python) 来处理表格数据。在本文中，我们将讨论如何在 python 中合并两个数据帧。

## Python 中如何合并两个数据帧？

假设我们有一个数据帧，其中包含一些学生的姓名、他们的学号以及他们选择学习的班级，如下所示。

```py
 Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
```

此外，我们有一个数据帧，其中包含学生的编号和他们获得的分数，如下所示。

```py
 Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
```

现在，我们必须合并给定的数据帧，使得结果数据帧包含每个学生的姓名、编号、年级、班级和相应的分数，如下所示。

```py
 Roll  Marks Grade  Class      Name
0    11     85     A      1    Aditya
1    12     95     A      1     Chris
2    13     75     B      1       Sam
3    14     75     B      1      Joel
4    16     78     B      1  Samantha
5    15     55     C      1       Tom
6    20     72     B      1      Tina
7    24     92     A      1       Amy
```

为了获得输出，我们将使用 pandas 模块中定义的`merge()`方法。`merge()`方法的语法如下。

```py
pd.merge(df1,df2, on) 
```

在这里，

*   `df1`表示第一个数据帧。
*   参数`df2`表示将被合并的第二数据帧。
*   参数`‘on’`是数据帧的列名，用于比较给定数据帧的列。如果两个数据帧中的行在对应于“on”参数的列中具有相同的值，则将它们合并在一起。

`merge()`方法还接受其他几个参数。你可以在这个[文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)里看看。

为了合并给定的数据帧，我们将调用`merge()`方法，首先将数据帧作为第一个输入参数。随后，我们将把包含标记的数据帧作为第二个输入参数传递给`merge()`方法。对于 `‘on’`参数，我们将传递`‘Roll’`列名。这样，来自给定数据帧的对应于相同卷号的行将被合并，并且将产生如下所示的结果数据帧。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,left_on="Roll", right_on="Roll")
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
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
   Roll  Marks Grade  Class      Name
0    11     85     A      1    Aditya
1    12     95     A      1     Chris
2    13     75     B      1       Sam
3    14     75     B      1      Joel
4    16     78     B      1  Samantha
5    15     55     C      1       Tom
6    20     72     B      1      Tina
7    24     92     A      1       Amy
```

如果第二个数据帧中的行与第一个数据帧中的任何行都不对应，则从结果中忽略这些行。类似地，如果第一个数据帧包含与第二个数据帧不对应的行，则这些行将从结果中省略。你可以在上面的例子中观察到这一点。

## 在 Python 中使用外部连接合并数据帧

为了包含省略的行，我们可以在`merge()`方法中使用参数`‘how’`。`‘how’`参数有默认值`‘inner’`。因此，只有那些同时出现在两个输入数据帧中的行才会包含在结果数据帧中。要包含省略的数据帧，您可以将值`‘outer’`传递给参数`‘how’`，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,how="outer",left_on="Roll", right_on="Roll",suffixes=("_left","_right"))
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
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
    Roll  Marks Grade  Class      Name
0     11   85.0     A    1.0    Aditya
1     12   95.0     A    1.0     Chris
2     13   75.0     B    1.0       Sam
3     14   75.0     B    1.0      Joel
4     16   78.0     B    1.0  Samantha
5     15   55.0     C    1.0       Tom
6     19   75.0     B    NaN       NaN
7     20   72.0     B    1.0      Tina
8     24   92.0     A    1.0       Amy
9     25   95.0     A    NaN       NaN
10    17    NaN   NaN    1.0     Pablo
11    30    NaN   NaN    1.0    Justin
12    31    NaN   NaN    1.0      Karl
```

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## 在 Python 中使用左右连接合并数据帧

如果您想只包含第一个数据帧中省略的行，可以将值`‘left’`传递给参数`‘how’`。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,how="left",left_on="Roll", right_on="Roll",suffixes=("_left","_right"))
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
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
   Roll  Marks Grade  Class      Name
0    11     85     A    1.0    Aditya
1    12     95     A    1.0     Chris
2    13     75     B    1.0       Sam
3    14     75     B    1.0      Joel
4    16     78     B    1.0  Samantha
5    15     55     C    1.0       Tom
6    19     75     B    NaN       NaN
7    20     72     B    1.0      Tina
8    24     92     A    1.0       Amy
9    25     95     A    NaN       NaN
```

类似地，如果您想只包含第二个数据帧中省略的行，您可以将值`‘right’`传递给参数`‘how’`，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,how="right",left_on="Roll", right_on="Roll",suffixes=("_left","_right"))
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
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
    Roll  Marks Grade  Class      Name
0     11   85.0     A      1    Aditya
1     12   95.0     A      1     Chris
2     13   75.0     B      1       Sam
3     14   75.0     B      1      Joel
4     15   55.0     C      1       Tom
5     16   78.0     B      1  Samantha
6     17    NaN   NaN      1     Pablo
7     20   72.0     B      1      Tina
8     24   92.0     A      1       Amy
9     30    NaN   NaN      1    Justin
10    31    NaN   NaN      1      Karl
```

## 结论

在本文中，我们讨论了如何使用`merge()` 方法在 python 中合并两个数据帧。要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[字典理解](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python)的文章。你可能也会喜欢这篇关于 python 中的[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)的文章。