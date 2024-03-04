# 检查熊猫 Python 中的 NaN 值

> 原文：<https://www.pythonforbeginners.com/basics/check-for-nan-values-in-pandas-python>

在 python 中处理数据时，我们经常会遇到 null 值或 NaN 值。在本文中，我们将讨论在 pandas 数据帧或系列中检查 nan 值或 null 值的不同方法。

## isna()函数

pandas 中的 isna()函数用于检查 NaN 值。它具有以下语法。

```py
pandas.isna(object)
```

这里，`object`可以是单个 python 对象，也可以是 python 对象的列表/数组。

如果我们将一个 python 对象作为输入参数传递给 `isna()`方法，如果 python 对象是 None，pd，它将返回 True。NA 或 np。NaN 对象。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=pd.NA
print("The value is:",x)
output=pd.isna(x)
print("Is the value Null:",output)
```

输出:

```py
The value is: <NA>
Is the value Null: True
```

在上面的例子中，我们已经超过了熊猫。NA 对象到`isna()` 函数。执行后，该函数返回 True。

当我们将元素的列表或 numpy 数组传递给 `isna()`函数时，`isna()`函数会对数组中的每个元素执行。

执行后，它返回一个包含真值和假值的列表或数组。输出数组的 False 值对应于在输入列表或数组中相同位置上不是 NA、NaN 或 None 的所有值。输出数组中的真值对应于输入列表或数组中相同位置的所有 NA、NaN 或 None 值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=[1,2,pd.NA,4,5,None, 6,7,np.nan]
print("The values are:",x)
output=pd.isna(x)
print("Are the values Null:",output)
```

输出:

```py
The values are: [1, 2, <NA>, 4, 5, None, 6, 7, nan]
Are the values Null: [False False  True False False  True False False  True]
```

在这个例子中，我们将一个包含 9 个元素的列表传递给了`isna()`函数。执行后，`isna()`方法返回一个包含 9 个布尔值的列表。输出列表中的每个元素都与给`isna()` 函数的输入列表中相同索引处的元素相关联。在输入列表包含空值的索引处，输出列表包含 True。类似地，在输入列表包含整数的索引处，输出列表包含 False。

## 使用 isna()方法检查 Pandas 数据帧中的 NaN 值

除了`isna()`函数，pandas 模块在数据帧级别也有 `isna()` 方法。您可以直接调用[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)上的`isna()`方法来检查 nan 值。

当在 pandas 数据帧上调用`isna()`方法时，它返回另一个包含真值和假值的数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe is:")
print(df)
output=df.isna()
print("Are the values Null:")
print(output)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
0       1    11      Aditya   85.0     A
1       1    12       Chris    NaN     A
2       1    14         Sam   75.0     B
3       1    15       Harry    NaN   NaN
4       2    22         Tom   73.0     B
5       2    15        Golu   79.0     B
6       2    27       Harsh   55.0     C
7       2    23       Clara    NaN     B
8       3    34         Amy   88.0     A
9       3    15    Prashant    NaN     B
10      3    27      Aditya   55.0     C
11      3    23  Radheshyam    NaN   NaN
Are the values Null:
    Class   Roll   Name  Marks  Grade
0   False  False  False  False  False
1   False  False  False   True  False
2   False  False  False  False  False
3   False  False  False   True   True
4   False  False  False  False  False
5   False  False  False  False  False
6   False  False  False  False  False
7   False  False  False   True  False
8   False  False  False  False  False
9   False  False  False   True  False
10  False  False  False  False  False
11  False  False  False   True   True
```

在上面的例子中，我们传递了一个包含 NaN 值和其他值的 dataframe。`isna()`方法返回包含布尔值的数据帧。这里，输出数据帧的假值对应于在输入数据帧中相同位置不是 NA、NaN 或 None 的所有值。输出数据帧中的真值对应于输入数据帧中相同位置的所有 NA、NaN 或 None 值。

## 检查熊猫数据框中某列的 Nan 值

除了整个数据帧，您还可以在 pandas 数据帧的一列中检查 nan 值。为此，您只需要调用特定列上的`isna()` 方法，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe column is:")
print(df["Marks"])
output=df["Marks"].isna()
print("Are the values Null:")
print(output)
```

输出:

```py
The dataframe column is:
0     85.0
1      NaN
2     75.0
3      NaN
4     73.0
5     79.0
6     55.0
7      NaN
8     88.0
9      NaN
10    55.0
11     NaN
Name: Marks, dtype: float64
Are the values Null:
0     False
1      True
2     False
3      True
4     False
5     False
6     False
7      True
8     False
9      True
10    False
11     True
Name: Marks, dtype: bool
```

## 使用 isna()方法检查熊猫系列中的 Nan 值

像 dataframe 一样，我们也可以对 pandas 中的 Series 对象调用`isna()` 方法。在这种情况下，`isna()`方法返回一个包含真值和假值的序列。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=pd.Series([1,2,pd.NA,4,5,None, 6,7,np.nan])
print("The series is:")
print(x)
output=pd.isna(x)
print("Are the values Null:")
print(output)
```

输出:

```py
The series is:
0       1
1       2
2    <NA>
3       4
4       5
5    None
6       6
7       7
8     NaN
dtype: object
Are the values Null:
0    False
1    False
2     True
3    False
4    False
5     True
6    False
7    False
8     True
dtype: bool
```

在这个例子中，我们在[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)上调用了`isna()` 方法。`isna()` 方法在执行后返回一系列布尔值。这里，输出序列的假值对应于在输入序列中相同位置不是 NA、NaN 或 None 的所有值。输出序列中的真值对应于输入序列中相同位置的所有 NA、NaN 或 None 值。

## 使用 isnull()方法检查熊猫的 NaN 值

`isnull()` 函数是 `isna()` 函数的别名。因此，它的工作方式与 `isna()` 功能完全相同。

当我们传递一个 NaN 值时，熊猫。NA 值，熊猫。NaT 值，或者 None 对象为`isnull()` 函数，则返回 True。

```py
import pandas as pd
import numpy as np
x=pd.NA
print("The value is:",x)
output=pd.isnull(x)
print("Is the value Null:",output)
```

输出:

```py
The value is: <NA>
Is the value Null: True
```

在上面的例子中，我们已经过了熊猫。`isnull()`功能的 NA 值。因此，它返回 True。

当我们将任何其他 python 对象传递给`isnull()`函数时，它返回 False，如下所示。

```py
import pandas as pd
import numpy as np
x=1117
print("The value is:",x)
output=pd.isnull(x)
print("Is the value Null:",output)
```

输出:

```py
The value is: 1117
Is the value Null: False
```

在这个例子中，我们将值 1117 传递给了`isnull()`函数。因此，它返回 False，表明该值不是空值。

当我们将一个列表或 numpy 数组传递给`isnull()` 函数时，它返回一个包含 True 和 False 值的 numpy 数组。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=[1,2,pd.NA,4,5,None, 6,7,np.nan]
print("The values are:",x)
output=pd.isnull(x)
print("Are the values Null:",output)
```

输出:

```py
The values are: [1, 2, <NA>, 4, 5, None, 6, 7, nan]
Are the values Null: [False False  True False False  True False False  True]
```

在这个例子中，我们将一个列表传递给了`isnull()` 函数。执行后， `isnull()` 函数返回一个布尔值列表。输出列表中的每个元素都与给`isnull()` 函数的输入列表中相同索引处的元素相关联。在输入列表包含空值的索引处，输出列表包含 True。类似地，在输入列表包含整数的索引处，输出列表包含 False。

## 使用 isnull()方法检查数据帧中的 NaN 值

您还可以调用 pandas 数据帧上的`isnull()` 方法来检查 nan 值，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe is:")
print(df)
output=df.isnull()
print("Are the values Null:")
print(output)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
0       1    11      Aditya   85.0     A
1       1    12       Chris    NaN     A
2       1    14         Sam   75.0     B
3       1    15       Harry    NaN   NaN
4       2    22         Tom   73.0     B
5       2    15        Golu   79.0     B
6       2    27       Harsh   55.0     C
7       2    23       Clara    NaN     B
8       3    34         Amy   88.0     A
9       3    15    Prashant    NaN     B
10      3    27      Aditya   55.0     C
11      3    23  Radheshyam    NaN   NaN
Are the values Null:
    Class   Roll   Name  Marks  Grade
0   False  False  False  False  False
1   False  False  False   True  False
2   False  False  False  False  False
3   False  False  False   True   True
4   False  False  False  False  False
5   False  False  False  False  False
6   False  False  False  False  False
7   False  False  False   True  False
8   False  False  False  False  False
9   False  False  False   True  False
10  False  False  False  False  False
11  False  False  False   True   True
```

在输出中，您可以观察到`isnull()`方法的行为方式与`isna()` 方法完全相同。

## 使用 isnull()方法检查 Dataframe 中列的 NaN

除了整个数据帧，您还可以使用`isnull()`方法来检查列中的 nan 值，如下例所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe column is:")
print(df["Marks"])
output=df["Marks"].isnull()
print("Are the values Null:")
print(output)
```

输出:

```py
The dataframe column is:
0     85.0
1      NaN
2     75.0
3      NaN
4     73.0
5     79.0
6     55.0
7      NaN
8     88.0
9      NaN
10    55.0
11     NaN
Name: Marks, dtype: float64
Are the values Null:
0     False
1      True
2     False
3      True
4     False
5     False
6     False
7      True
8     False
9      True
10    False
11     True
Name: Marks, dtype: bool
```

以类似的方式，您可以对 pandas 系列调用 `isnull()`方法，如下所示。

```py
import pandas as pd
import numpy as np
x=pd.Series([1,2,pd.NA,4,5,None, 6,7,np.nan])
print("The series is:")
print(x)
output=pd.isnull(x)
print("Are the values Null:")
print(output)
```

输出:

```py
The series is:
0       1
1       2
2    <NA>
3       4
4       5
5    None
6       6
7       7
8     NaN
dtype: object
Are the values Null:
0    False
1    False
2     True
3    False
4    False
5     True
6    False
7    False
8     True
dtype: bool
```

在上面的例子中，我们对一个系列调用了`isnull()`方法。`isnull()`方法在执行后返回一系列布尔值。这里，输出序列的假值对应于在输入序列中相同位置不是 NA、NaN 或 None 的所有值。输出序列中的真值对应于输入序列中相同位置的所有 NA、NaN 或 None 值。

## 结论

在这篇文章中，我们讨论了检查熊猫的 nan 值的不同方法。要了解更多关于 python 编程的知识，

你可以阅读这篇关于如何对熊猫数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！