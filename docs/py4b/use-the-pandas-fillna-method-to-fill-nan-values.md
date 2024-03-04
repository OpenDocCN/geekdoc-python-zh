# 使用 Pandas fillna 方法填充 NaN 值

> 原文：<https://www.pythonforbeginners.com/basics/use-the-pandas-fillna-method-to-fill-nan-values>

在分析数据时处理 NaN 值是一项重要的任务。python 中的 pandas 模块为我们提供了填充 NaN 值的`fillna()`方法。在本文中，我们将讨论如何使用 pandas fillna 方法在 Python 中填充 NaN 值。

## filna()方法

您可以使用`fillna()` 方法在 pandas 数据帧中填充 NaN 值。它具有以下语法。

```py
DataFrame.fillna(value=None, *, method=None, axis=None, inplace=False, limit=None, downcast=None)
```

这里，

*   `value`参数取替换 NaN 值的值。还可以将一个 [python 字典](https://www.pythonforbeginners.com/dictionary/how-to-use-dictionaries-in-python/)或一个序列传递给 value 参数。这里，字典应该包含作为键的数据帧的列名，以及作为关联值需要填充到列中的值。类似地， [pandas series](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python) 应该包含数据帧的列名作为索引，包含替换值作为每个索引的关联值。
*   如果没有输入给`value`参数，则`method`参数用于填充数据帧中的 NaN 值。如果`value`参数不为无，则`method`参数被设置为无。否则，我们可以指定文字`“ffill”`、`“bfill”`、`“backfill”`或 `“pad”`来指定我们想要填充什么值来代替 NaN 值。
*   `axis`参数用于指定填充缺失值的轴。如果您想使用 pandas fillna 方法只填充特定的行或列，您可以使用`axis`参数。为了填充行中的 NaN 值，`axis`参数被设置为 1 或`“columns”`。为了将值填充到列中，`axis`参数被设置为`“index”`或 0。
*   默认情况下，pandas fillna 方法不修改原始数据帧。它在执行后返回一个新的数据帧，要修改调用`fillna()` 方法的原始数据帧，可以将`inplace`参数设置为 True。
*   如果指定了`method`参数，则`limit`参数指定向前/向后填充的连续 NaN 值的最大数量。换句话说，如果连续的“南”数超过了`limit`数，那么这个缺口只能被部分填补。如果未指定`method`参数，则`limit`参数取沿整个轴的最大条目数，其中 nan 将被填充。如果不是无，它必须大于 0。
*   如果需要更改值的数据类型，参数`downcast`将字典作为映射来决定应该向下转换的数据类型和目标数据类型。

## 使用 Pandas Fillna 填充整个数据帧中的 Nan 值

要使用 fillna 方法填充 pandas 数据帧中的 NaN 值，需要将 NaN 值的替换值传递给`fillna()`方法，如下例所示。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x=x.fillna(0)
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0           0    0.0     0
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0           0   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     0.0   0.0           0    0.0     0
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0     0
9     0.0   0.0           0    0.0     0
10    3.0  15.0      Lokesh   88.0     A
```

在上面的例子中，我们将值 0 传递给了`fillna()`方法。因此，输入数据帧中的所有 NaN 值都被替换为 0。

这种方法不太实际，因为不同的列有不同的数据类型。因此，我们可以选择在不同的列中填充不同的值来替换空值。

## 在熊猫的每一列中填入不同的值

除了用相同的值填充所有 NaN 值，还可以用特定的值替换每列中的 NaN 值。为此，我们需要向`fillna()`方法传递一个字典，该字典包含作为键的列名和作为关联值填充到列中的值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x=x.fillna({"Class":1,"Roll":100,"Name":"PFB","Marks":0,"Grade":"F"})
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
    Class   Roll        Name  Marks Grade
0     2.0   27.0       Harsh   55.0     C
1     2.0   23.0       Clara   78.0     B
2     3.0   33.0         PFB    0.0     F
3     3.0   34.0         Amy   88.0     A
4     3.0   15.0         PFB   78.0     B
5     3.0   27.0      Aditya   55.0     C
6     1.0  100.0         PFB    0.0     F
7     3.0   23.0  Radheshyam   78.0     B
8     3.0   11.0       Bobby   50.0     F
9     1.0  100.0         PFB    0.0     F
10    3.0   15.0      Lokesh   88.0     A
```

在上面的例子中，我们将字典`{"Class" :1, "Roll": 100, "Name": "PFB", "Marks" : 0, "Grade": "F" }`作为输入传递给了`fillna()` 方法。因此，`"Class"`列中的 NaN 值被替换为 1，`"Roll"`列中的 NaN 值被替换为 100，`"Name"`列中的 NaN 值被替换为`"PFB"`，依此类推。因此，当我们将数据帧的列名作为键并将一个 [python 文字](https://www.pythonforbeginners.com/basics/python-literals)作为关联值传递给键时，NaN 值将根据输入字典在数据帧的每一列中被替换。

您也可以选择忽略一些列名，而不是将所有列名作为输入字典中的键。在这种情况下，不考虑替换输入字典中不存在的列中的 NaN 值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x=x.fillna({"Class":1,"Roll":100,"Name":"PFB","Marks":0})
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
    Class   Roll        Name  Marks Grade
0     2.0   27.0       Harsh   55.0     C
1     2.0   23.0       Clara   78.0     B
2     3.0   33.0         PFB    0.0   NaN
3     3.0   34.0         Amy   88.0     A
4     3.0   15.0         PFB   78.0     B
5     3.0   27.0      Aditya   55.0     C
6     1.0  100.0         PFB    0.0   NaN
7     3.0   23.0  Radheshyam   78.0     B
8     3.0   11.0       Bobby   50.0   NaN
9     1.0  100.0         PFB    0.0   NaN
10    3.0   15.0      Lokesh   88.0     A
```

在这个例子中，我们没有将输入字典中的`"Grade"`列传递给`fillna()`方法。因此，`"Grade"`列中的 NaN 值不会被任何其他值替换。

## 仅填充每列中的前 N 个空值

您也可以限制每列中要填充的 NaN 值的数量，而不是在每列中填充所有 NaN 值。为此，您可以将最大数量的值作为输入参数传递给`fillna()`方法中的`limit`参数，如下所示。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x=x.fillna(0, limit=3)
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0           0    0.0     0
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0           0   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     0.0   0.0           0    0.0     0
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0     0
9     0.0   0.0         NaN    0.0   NaN
10    3.0  15.0      Lokesh   88.0     A
```

在上面的例子中，我们已经将`limit`参数设置为 3。因此，只有每列的前三个 NaN 值被替换为 0。

## 仅填充每行的前 N 个空值

要仅填充 dataframe 每一行中的前 N 个空值，可以将要填充的最大值作为输入参数传递给 fillna()方法中的`limit`参数。此外，您需要通过将`axis`参数设置为 1 来指定您想要填充的行。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x=x.fillna(0, limit=2,axis=1)
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
   Class  Roll        Name Marks Grade
0    2.0  27.0       Harsh  55.0     C
1    2.0  23.0       Clara  78.0     B
2    3.0  33.0         0.0   0.0   NaN
3    3.0  34.0         Amy  88.0     A
4    3.0  15.0           0  78.0     B
5    3.0  27.0      Aditya  55.0     C
6    0.0   0.0         NaN   NaN   NaN
7    3.0  23.0  Radheshyam  78.0     B
8    3.0  11.0       Bobby  50.0     0
9    0.0   0.0         NaN   NaN   NaN
10   3.0  15.0      Lokesh  88.0     A
```

在上面的例子中，我们将`limit`参数设置为 2，将`axis`参数设置为 1。因此，当执行`fillna()`方法时，每行只有两个 NaN 值被替换为 0。

## 熊猫 Fillna 与最后一个有效的观察

您也可以使用现有值填充 NaN 值，而不是指定新值。例如，您可以通过将方法参数设置为如下所示的`“ffill”`，使用最后一次有效观察来填充空值。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x=x.fillna(method="ffill")
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0       Clara   78.0     B
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         Amy   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     3.0  27.0      Aditya   55.0     C
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0     B
9     3.0  11.0       Bobby   50.0     B
10    3.0  15.0      Lokesh   88.0     A
```

在本例中，我们将方法参数设置为`"ffill"`。因此，每当遇到 NaN 值时，`fillna()`方法就用同一列中前一个单元格中的非空值填充特定的单元格。

## 熊猫 Fillna 与下一个有效的观察

您可以通过将`method`参数设置为`“bfill”` 来使用下一个有效观测值填充空值，如下所示。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x=x.fillna(method="bfill")
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         Amy   88.0     A
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0      Aditya   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     3.0  23.0  Radheshyam   78.0     B
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0     A
9     3.0  15.0      Lokesh   88.0     A
10    3.0  15.0      Lokesh   88.0     A
```

在本例中，我们已经将`method`参数设置为`"bfill"`。因此，每当遇到 NaN 值时，`fillna()`方法就用同一列中下一个单元格中的非空值填充特定的单元格。

## 熊猫在原地飞

默认情况下，`fillna()`方法在执行后返回一个新的数据帧。要修改现有的数据帧而不是创建一个新的数据帧，可以在如下所示的`fillna()`方法中将`inplace`参数设置为 True。

```py
import pandas as pd
import numpy as np
x=pd.read_csv("grade2.csv")
print("The original dataframe is:")
print(x)
x.fillna(method="bfill",inplace=True)
print("The modified dataframe is:")
print(x)
```

输出:

```py
The original dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     NaN   NaN         NaN    NaN   NaN
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
9     NaN   NaN         NaN    NaN   NaN
10    3.0  15.0      Lokesh   88.0     A
The modified dataframe is:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         Amy   88.0     A
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0      Aditya   78.0     B
5     3.0  27.0      Aditya   55.0     C
6     3.0  23.0  Radheshyam   78.0     B
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0     A
9     3.0  15.0      Lokesh   88.0     A
10    3.0  15.0      Lokesh   88.0     A
```

在这个例子中，我们已经在`fillna()`方法中将 inplace 参数设置为 True。因此，输入数据帧被修改。

## 结论

在本文中，我们讨论了如何使用 pandas fillna 方法在 Python 中填充 nan 值。

要了解更多关于 python 编程的知识，你可以阅读这篇关于如何对熊猫数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！