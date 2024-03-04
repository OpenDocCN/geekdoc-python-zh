# 检查熊猫 Python 中的非空值

> 原文：<https://www.pythonforbeginners.com/basics/check-for-not-null-value-in-pandas-python>

在 python 中，我们有时需要过滤非空值和空值。在本文中，我们将使用示例讨论在 pandas 中检查 not null 的不同方法。

我们可以使用`notna()` 函数和 `notnull()`函数在 pandas 中检查 not null。让我们逐一讨论每个功能。

## 使用 notna()方法检查 Pandas 中的 Not Null

顾名思义，`notna()`方法是对`isna()`方法的否定。`isna()`方法用于[检查熊猫](https://www.pythonforbeginners.com/basics/check-for-nan-values-in-pandas-python)的 nan 值。`notna()` 函数的语法如下。

```py
pandas.notna(object)
```

这里，`object`可以是单个 python 对象，也可以是对象的集合，比如 [python 列表](https://www.pythonforbeginners.com/lists/python-lists-cheat-sheet-2)或元组。

如果我们将一个 python 对象作为输入参数传递给`notna()` 方法，如果 python 对象是 None，pd，它将返回 False。NA 或 np。NaN 对象。对于不为空的 python 对象，`notna()`函数返回 True。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=pd.NA
print("The value is:",x)
output=pd.notna(x)
print("Is the value not Null:",output)
```

输出:

```py
The value is: <NA>
Is the value not Null: False
```

在上面的例子中，我们已经超过了熊猫。NA 对象到`notna()`函数。因此，它返回 False。

当我们将元素的列表或 numpy 数组传递给`notna()` 函数时，`notna()`函数会对数组中的每个元素执行。执行后，它返回一个包含真值和假值的列表或数组。输出数组的真值对应于输入列表或数组中相同位置上所有不为 NA、NaN 或 None 的值。输出数组中的 False 值对应于输入列表或数组中相同位置的所有 NA、NaN 或 None 值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=[1,2,pd.NA,4,5,None, 6,7,np.nan]
print("The values are:",x)
output=pd.notna(x)
print("Are the values not Null:",output)
```

输出:

```py
The values are: [1, 2, <NA>, 4, 5, None, 6, 7, nan]
Are the values not Null: [ True  True False  True  True False  True  True False]
```

在这个例子中，我们将一个包含 9 个元素的列表传递给了`notna()` 函数。执行后，`notna()` 函数返回一个包含 9 个布尔值的列表。输出列表中的每个元素都与给`notna()`函数的输入列表中相同索引处的元素相关联。在输入列表不包含空值的索引处，输出列表包含 True。类似地，在输入列表包含空值的索引处，输出列表包含 False。

## 使用 notna()方法检查熊猫数据帧中的 Not NA

除了`notna()`函数，python 还为我们提供了`notna()`方法来检查 pandas 数据帧和系列对象中的非空值。

当在[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)上调用`notna()` 方法时，返回另一个包含真值和假值的数据帧。输出数据帧的真值对应于在输入数据帧中相同位置不是 NA、NaN 或 None 的所有值。输出数据帧中的假值对应于输入数据帧中相同位置的所有 NA、NaN 或 None 值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe is:")
print(df)
output=df.notna()
print("Are the values not Null:")
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
Are the values not Null:
    Class  Roll  Name  Marks  Grade
0    True  True  True   True   True
1    True  True  True  False   True
2    True  True  True   True   True
3    True  True  True  False  False
4    True  True  True   True   True
5    True  True  True   True   True
6    True  True  True   True   True
7    True  True  True  False   True
8    True  True  True   True   True
9    True  True  True  False   True
10   True  True  True   True   True
11   True  True  True  False  False
```

在上面的例子中，我们在包含 NaN 值和其他值的数据帧上调用了`notna()` 方法。`notna()`方法返回一个包含布尔值的数据帧。这里，输出数据帧的假值对应于在输入数据帧中相同位置为 NA、NaN 或 None 的所有值。输出数据帧中的真值对应于输入数据帧中相同位置的所有非空值。

## 检查 Pandas 数据帧中某列的非空值

除了整个数据帧，您还可以在 pandas 数据帧的列中检查 not null 值。为此，您只需要调用特定列上的`notna()`方法，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe column is:")
print(df["Marks"])
output=df["Marks"].notna()
print("Are the values not Null:")
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
Are the values not Null:
0      True
1     False
2      True
3     False
4      True
5      True
6      True
7     False
8      True
9     False
10     True
11    False
Name: Marks, dtype: bool
```

## 使用 notna()方法检查熊猫系列中的 Not NA

像 dataframe 一样，我们也可以在一个[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)对象上调用`notna()` 方法。在这种情况下，`notna()`方法返回一个包含真值和假值的序列。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=pd.Series([1,2,pd.NA,4,5,None, 6,7,np.nan])
print("The series is:")
print(x)
output=pd.notna(x)
print("Are the values not Null:")
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
Are the values not Null:
0     True
1     True
2    False
3     True
4     True
5    False
6     True
7     True
8    False
dtype: bool
```

在这个例子中，我们对熊猫系列调用了`notna()`方法。`notna()`方法在执行后返回一系列布尔值。这里，输出序列的假值对应于在输入序列中相同位置为 NA、NaN 或 None 的所有值。输出序列中的真值对应于输入序列中相同位置的所有非空值。

## 使用 notnull()方法检查 Pandas 中的 Not Null

`notnull()`方法是`notna()`方法的别名。因此，它的工作原理与`notna()`方法完全相同。

当我们传递一个 NaN 值时，熊猫。NA 值，熊猫。NaT 值，或者 None 对象为`notnull()` 函数，则返回 False。

```py
import pandas as pd
import numpy as np
x=pd.NA
print("The value is:",x)
output=pd.notnull(x)
print("Is the value not Null:",output)
```

输出:

```py
The value is: <NA>
Is the value not Null: False
```

在上面的例子中，我们已经过了熊猫。`notnull()`功能的 NA 值。因此，它返回 False。

当我们将任何其他 python 对象传递给`notnull()` 函数时，它返回 True，如下所示。

```py
import pandas as pd
import numpy as np
x=1117
print("The value is:",x)
output=pd.notnull(x)
print("Is the value not Null:",output)
```

输出:

```py
The value is: 1117
Is the value not Null: True
```

在这个例子中，我们将值 1117 传递给了`notnull()`函数。因此，它返回 True，表明该值不是空值。

当我们将一个列表或 numpy 数组传递给`notnull()`函数时，它返回一个包含 True 和 False 值的 numpy 数组。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=[1,2,pd.NA,4,5,None, 6,7,np.nan]
print("The values are:",x)
output=pd.notnull(x)
print("Are the values not Null:",output)
```

输出:

```py
The values are: [1, 2, <NA>, 4, 5, None, 6, 7, nan]
Are the values not Null: [ True  True False  True  True False  True  True False]
```

在这个例子中，我们将一个列表传递给了`notnull()` 函数。执行后，`notnull()` 函数返回一个布尔值列表。输出列表中的每个元素都与输入列表中相同索引处的元素相关联，该索引被赋予`notnull()`功能。在输入列表包含 Null 值的索引处，输出列表包含 False。类似地，在输入列表包含整数的索引处，输出列表包含 True。

## 使用 notnull()方法检查 Pandas 数据帧中的 Not Null

您还可以调用 pandas 数据帧上的`notnull()`方法来检查 nan 值，如下所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe is:")
print(df)
output=df.notnull()
print("Are the values not Null:")
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
Are the values not Null:
    Class  Roll  Name  Marks  Grade
0    True  True  True   True   True
1    True  True  True  False   True
2    True  True  True   True   True
3    True  True  True  False  False
4    True  True  True   True   True
5    True  True  True   True   True
6    True  True  True   True   True
7    True  True  True  False   True
8    True  True  True   True   True
9    True  True  True  False   True
10   True  True  True   True   True
11   True  True  True  False  False
```

在输出中，您可以观察到`notn` ull()方法的行为方式与`notna()`方法完全相同。

除了整个 dataframe，您还可以使用`notnull()`方法来检查列中的 not nan 值，如下例所示。

```py
import pandas as pd
import numpy as np
df=pd.read_csv("grade.csv")
print("The dataframe column is:")
print(df["Marks"])
output=df["Marks"].notnull()
print("Are the values not Null:")
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
Are the values not Null:
0      True
1     False
2      True
3     False
4      True
5      True
6      True
7     False
8      True
9     False
10     True
11    False
Name: Marks, dtype: bool
```

以类似的方式，您可以对 pandas 系列调用`notnull()` 方法，如下所示。

```py
import pandas as pd
import numpy as np
x=pd.Series([1,2,pd.NA,4,5,None, 6,7,np.nan])
print("The series is:")
print(x)
output=pd.notnull(x)
print("Are the values not Null:")
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
Are the values not Null:
0     True
1     True
2    False
3     True
4     True
5    False
6     True
7     True
8    False
dtype: bool
```

在上面的例子中，我们对一个系列调用了`notnull()`方法。`notnull()`方法在执行后返回一系列布尔值。这里，输出序列的真值对应于在输入序列中相同位置不是 NA、NaN 或 None 的所有值。输出序列中的假值对应于输入序列中相同位置的所有 NA、NaN 或 None 值。

## 结论

在本文中，我们讨论了在 pandas 中检查非空值的不同方法。要了解更多关于 python 编程的知识，你可以阅读这篇关于如何对熊猫数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！