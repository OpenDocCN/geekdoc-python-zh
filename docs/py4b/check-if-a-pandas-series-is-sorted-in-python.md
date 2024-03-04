# 检查熊猫系列是否在 Python 中排序

> 原文：<https://www.pythonforbeginners.com/basics/check-if-a-pandas-series-is-sorted-in-python>

Pandas 系列是在 python 中处理顺序数据的一个很好的工具。在本文中，我们将讨论检查熊猫系列是否排序的不同方法。

## 检查熊猫系列是否使用 is_monotonic 属性排序

要检查一个[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)是否按升序排序，我们可以使用该系列的`is_monotonic`属性。如果熊猫系列按升序排序，则属性`is_monotonic`的计算结果为 True。

例如，如果一个序列是按升序排序的，那么`is_monotonic`属性的值将为 True，如下所示。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic)
```

输出:

```py
The series is:
2   -100
5     -3
0      3
3     14
4     16
1     23
6     45
7     65
dtype: int64
The series is sorted?:
True
```

在上面的例子中，我们首先使用`Series()`构造函数创建了一个系列。然后，我们使用`sort_values()`方法对序列进行排序。在这之后，我们调用了这个系列的`is_monotonic`属性。您可以观察到，在按升序排序的序列上调用时，`is_monotonic`属性的计算结果为 True。

如果序列按降序排序，`is_monotonic`属性的计算结果将为 False。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True,ascending=False)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic)
```

输出:

```py
The series is:
7     65
6     45
1     23
4     16
3     14
0      3
5     -3
2   -100
dtype: int64
The series is sorted?:
False
```

在这个例子中，我们已经在`sort_values()`方法中将`ascending`参数设置为 False。因此，该系列按降序排序。当我们调用按降序排序的序列的`is_monotonic`属性时，它的计算结果为 False。

如果熊猫系列没有排序，`is_monotonic`属性将被评估为假。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic)
```

输出:

```py
The series is:
0      3
1     23
2   -100
3     14
4     16
5     -3
6     45
7     65
dtype: int64
The series is sorted?:
False
```

在上面的例子中，我们没有对序列进行排序。因此，`is_monotonic`属性的计算结果为 False。

`is_monotonic`属性不适用于 NaN 值。如果一个序列包含 NaN 值，`is_monotonic`属性的计算结果总是为 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic)
```

输出:

```py
The series is:
0     3.0
3    14.0
4    16.0
1    23.0
6    45.0
7    65.0
2     NaN
5     NaN
dtype: float64
The series is sorted?:
False
```

在上面的例子中，我们创建了一个具有 NaN 值的序列。然后，我们按升序对系列进行排序。当对具有 NaN 值的排序序列调用`is_monotonic`属性时，其计算结果为 False，如输出所示。有人可能会说，排序后的序列中的 NaN 值在底部。这就是为什么`is_monotonic`属性的计算结果为 False。

然而，即使我们将 NaN 值放在熊猫系列的顶部，`is_monotonic`属性也将计算为 False，如下所示。

```py
import pandas as pd
import numpy as np
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True,na_position="first")
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic)
```

输出:

```py
The series is:
2     NaN
5     NaN
0     3.0
3    14.0
4    16.0
1    23.0
6    45.0
7    65.0
dtype: float64
The series is sorted?:
False
```

在上面的示例中，NaN 值出现在排序序列的顶部。即使在这之后，`is_monotonic`属性的计算结果也是 False。

使用`is_monotonic`属性时，您将得到一个 FutureWarning，并显示消息“ **FutureWarning: is_monotonic 已被否决，将在未来版本中删除。请改用 is_monotonic_increasing。**”这意味着`is_monotonic`属性将在未来的 pandas 版本中被弃用。作为替代，我们可以使用`is_monotonic_increasing`和`is_monotonic_decreasing`属性来检查熊猫系列在 python 中是否排序。

## 检查熊猫系列是否按升序排序

要检查熊猫系列是否按升序排序，我们可以使用`is_monotonic_increasing`属性。如果序列按升序排序，则`is_monotonic_increasing`属性的计算结果为 True。否则，它被设置为 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_increasing)
```

输出:

```py
The series is:
2   -100
5     -3
0      3
3     14
4     16
1     23
6     45
7     65
dtype: int64
The series is sorted?:
True
```

在上面的例子中，我们使用了`is_monotonic_increasing`属性来检查序列是否被排序。因为我们已经按升序对系列进行了排序，所以`is_monotonic_increasing`属性的计算结果为 True。

如果序列没有按升序排序，则`is_monotonic_increasing`属性的计算结果为 False。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_increasing)
```

输出:

```py
The series is:
0      3
1     23
2   -100
3     14
4     16
5     -3
6     45
7     65
dtype: int64
The series is sorted?:
False
```

在上面的例子中，我们没有对输入序列进行排序。因此，当在 series 对象上调用时，`is_monotonic_increasing`属性的计算结果为 False。

此外，如果熊猫系列按降序排序，`is_monotonic_increasing`属性的计算结果为 False。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True,ascending=False)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_increasing)
```

输出:

```py
The series is:
7     65
6     45
1     23
4     16
3     14
0      3
5     -3
2   -100
dtype: int64
The series is sorted?:
False
```

在本示例中，系列按降序排序。因此，`is_monotonic_increasing`属性的计算结果为 False。

`is_monotonic_increasing`不能用于具有 NaN 值的系列对象。如果 pandas 系列有 NaN 值，则`is_monotonic_increasing`属性总是计算为 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_increasing)
```

输出:

```py
The series is:
0     3.0
3    14.0
4    16.0
1    23.0
6    45.0
7    65.0
2     NaN
5     NaN
dtype: float64
The series is sorted?:
False
```

在上面的例子中，我们创建了一个具有 NaN 值的序列。然后，我们按升序对系列进行排序。当对具有 NaN 值的排序序列调用`is_monotonic_increasing`属性时，其计算结果为 False，如输出所示。有人可能会说，排序后的序列中的 NaN 值在底部。这就是为什么`is_monotonic_increasing`属性的计算结果为 False。

即使我们将 NaN 值放在序列的顶部,`is_monotonic_increasing`属性也将计算为 False。

```py
import pandas as pd
import numpy as np
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True,na_position="first")
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_increasing)
```

输出:

```py
The series is:
2     NaN
5     NaN
0     3.0
3    14.0
4    16.0
1    23.0
6    45.0
7    65.0
dtype: float64
The series is sorted?:
False
```

在本例中，NaN 值放在排序序列的顶部。即使在这之后，`is_monotonic_increasing`属性的计算结果也是 False。它显示了`is_monotonic_increasing`属性不支持 NaN 值。

有趣的阅读:[Python 中的命令行参数](https://avidpython.com/python-basics/command-line-argument-using-sys-argv-in-python/)

## 检查一个系列在 Python 中是否按降序排序

为了检查熊猫系列是否按降序排序，我们将使用`is_monotonic_decreasing`属性。如果序列按降序排序，则`is_monotonic_decreasing`属性的计算结果为 True。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True,ascending=False)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_decreasing)
```

输出:

```py
The series is:
7     65
6     45
1     23
4     16
3     14
0      3
5     -3
2   -100
dtype: int64
The series is sorted?:
True
```

在上面的例子中，我们使用了`is_monotonic_decreasing`属性来检查序列是否按降序排序。因为我们已经按升序对系列进行了排序，所以`is_monotonic_decreasing`属性的计算结果为 True。

如果序列未排序或按升序排序，则`is_monotonic_decreasing`属性的计算结果为 False，如下所示。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_decreasing)
```

输出:

```py
The series is:
0      3
1     23
2   -100
3     14
4     16
5     -3
6     45
7     65
dtype: int64
The series is sorted?:
False
```

在上面的例子中，`is_monotonic_decreasing`属性是在一个未排序的序列上调用的。因此，它的计算结果为假。

`is_monotonic_decreasing`属性不能用于具有 NaN 值的系列对象。如果 pandas 系列有 NaN 值，则`is_monotonic_decreasing`属性总是计算为 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True,ascending=False)
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_decreasing)
```

输出:

```py
The series is:
7    65.0
6    45.0
1    23.0
4    16.0
3    14.0
0     3.0
2     NaN
5     NaN
dtype: float64
The series is sorted?:
False 
```

在上面的例子中，我们创建了一个具有 NaN 值的序列。然后，我们按降序对系列进行排序。当对具有 NaN 值的排序序列调用`is_monotonic_decreasing`属性时，其计算结果为 False，如输出所示。有人可能会说，排序后的序列中的 NaN 值在底部。这就是为什么`is_monotonic_decreasing`属性的计算结果为 False。

然而，即使我们将 NaN 值放在序列的顶部，`is_monotonic_decreasing`属性也将计算为 False。

```py
import pandas as pd
import numpy as np
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(numbers)
series.sort_values(inplace=True,ascending=False,na_position="first")
print("The series is:")
print(series)
print("The series is sorted?:")
print(series.is_monotonic_decreasing)
```

输出:

```py
The series is:
2     NaN
5     NaN
7    65.0
6    45.0
1    23.0
4    16.0
3    14.0
0     3.0
dtype: float64
The series is sorted?:
False 
```

在本例中，我们将 NaN 值放在排序序列的顶部。即使在这之后，`is_monotonic_decreasing`属性的计算结果也是 False。这证实了我们可以将`is_monotonic_decreasing`用于具有 NaN 值的序列。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇 [MLFlow 教程，里面有代码示例](https://codinginfinite.com/mlflow-tutorial-with-code-example/)。您可能还会喜欢这篇关于用 Python 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## 检查序列是否使用 Numpy 模块排序

python 中的 numpy 模块为我们提供了不同的函数来对数值数据执行操作。一个这样的函数是`diff()` 函数。`diff()` 函数将一个像 series 这样的可迭代对象作为其输入参数，并返回一个包含数组元素的一阶差的数组，如下例所示。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
print("The series is:")
print(series)
temp=np.diff(series)
print("Array returned by diff() is:")
print(temp)
```

输出:

```py
The series is:
0      3
1     23
2   -100
3     14
4     16
5     -3
6     45
7     65
dtype: int64
Array returned by diff() is:
[  20 -123  114    2  -19   48   20]
```

在上面的例子中，我们已经将序列传递给了`diff()`函数。您可以观察到列表中的元素是系列中元素的差异。例如，输出列表中的第一个元素是序列中第二个和第一个元素之间的差。类似地，输出列表中的第二个元素是序列的第三和第二个元素之间的差。

因此，您可以观察到，一阶差被计算为输入序列中第(n+1)个和第 n 个元素之间的差。

为了检查熊猫系列是否使用`diff()`函数按升序排序，我们将使用以下步骤。

*   首先我们来计算熊猫系列的一阶差分。为此，我们将把序列作为输入参数传递给`diff()`函数。
*   之后，我们将检查输出数组中的所有元素是否都大于或等于 0。为此，我们将使用比较运算符和`all()`方法。当我们在 numpy 数组上使用比较运算符时，我们会得到一个布尔值数组。在包含布尔值的数组上调用`all()` 方法时，如果所有元素都为真，则返回真。
*   如果`all()` 方法返回 True，它将得出熊猫系列按升序排序的结论。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
series=series.sort_values()
print("The series is:")
print(series)
temp=np.diff(series)
boolean_array= temp>=0
print("Boolean array is:")
print(boolean_array)
result=boolean_array.all()
if result:
    print("The series is sorted.")
else:
    print("The series is not sorted.")
```

输出:

```py
The series is:
2   -100
5     -3
0      3
3     14
4     16
1     23
6     45
7     65
dtype: int64
Boolean array is:
[ True  True  True  True  True  True  True]
The series is sorted.
```

为了检查熊猫系列是否按降序排序，我们将检查`diff()`函数的输出数组中的所有元素是否都小于或等于 0。为此，我们将使用比较运算符和`all()`方法。当我们在 numpy 数组上使用比较运算符时，我们会得到一个布尔值数组。在包含布尔值的数组上调用`all()`方法时，如果所有元素都为真，则返回真。

如果`all()`方法返回 True，它将得出熊猫系列按降序排序的结论。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(numbers)
series=series.sort_values(ascending=False)
print("The series is:")
print(series)
temp=np.diff(series)
boolean_array= temp<=0
print("Boolean array is:")
print(boolean_array)
result=boolean_array.all()
if result:
    print("The series is sorted.")
else:
    print("The series is not sorted.")
```

输出:

```py
The series is:
7     65
6     45
1     23
4     16
3     14
0      3
5     -3
2   -100
dtype: int64
Boolean array is:
[ True  True  True  True  True  True  True]
The series is sorted.
```

## 检查 Python 中序列的索引是否排序

要检查熊猫系列的索引是否排序，我们可以使用如下所示的`index`属性和`is_monotonic`属性。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
sorted_series=series.sort_index()
print("The series is:")
print(sorted_series)
print("The index of series is sorted?:")
print(sorted_series.index.is_monotonic)
```

输出:

```py
The series is:
2     abcd
3        a
11       c
14      ab
16     abc
23       b
45      bc
65       d
dtype: object
The index of series is sorted?:
True
```

要检查熊猫系列的索引是否按升序排序，我们可以使用如下所示的`index`属性和`is_monotonic_increasing`属性。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
sorted_series=series.sort_index()
print("The series is:")
print(sorted_series)
print("The index of series is sorted?:")
print(sorted_series.index.is_monotonic_increasing)
```

输出:

```py
The series is:
2     abcd
3        a
11       c
14      ab
16     abc
23       b
45      bc
65       d
dtype: object
The index of series is sorted?:
True
```

要检查熊猫系列的索引是否按降序排序，我们可以使用如下所示的`index`属性和`is_monotonic_decreasing`属性。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
sorted_series=series.sort_index(ascending=False)
print("The series is:")
print(sorted_series)
print("The index of series is sorted?:")
print(sorted_series.index.is_monotonic_decreasing)
```

输出:

```py
The series is:
65       d
45      bc
23       b
16     abc
14      ab
11       c
3        a
2     abcd
dtype: object
The index of series is sorted?:
True
```

您需要记住，如果索引列包含 NaN 值，那么`is_monotonic`属性、`is_monotonic_increasing`属性和`is_monotonic_decreasing`总是返回 False。因此，如果索引列包含 NaN 值，则不能使用这些属性来检查索引是否已排序。

## 结论

在本文中，我们讨论了检查熊猫系列是否排序的不同方法。要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)