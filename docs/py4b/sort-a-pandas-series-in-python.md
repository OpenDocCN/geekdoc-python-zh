# 用 Python 对熊猫系列进行排序

> 原文：<https://www.pythonforbeginners.com/basics/sort-a-pandas-series-in-python>

Pandas 系列用于处理 python 中的顺序数据。在本文中，我们将讨论在 Python 中对熊猫系列进行排序的不同方法。

## 使用 sort_values()方法对序列进行排序

您可以使用`sort_values()`方法对熊猫系列进行排序。它具有以下语法。

```py
Series.sort_values(*, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)
```

这里，

*   `axis`参数用于决定我们是否要按列或行对数据帧进行排序。对于系列，不使用`axis`参数。定义它只是为了让`sort_values()`方法与[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)兼容。
*   默认情况下，`sort_values()`方法按升序对序列进行排序。如果您想按降序对一个系列进行排序，您可以将`ascending`参数设置为 True。
*   执行后，`sort_values()`方法返回排序后的序列。它不会修改原始系列。要对原始序列进行排序和修改，而不是创建一个新序列，您可以将`inplace`参数设置为`True`。
*   `kind`参数用于确定排序算法。默认情况下，使用`“quicksort”` 算法。如果您的数据具有特定的模式，而另一种排序算法可能是有效的，那么您可以使用`‘mergesort’`、 `‘heapsort’`或 `‘stable’` 排序算法。
*   `na_position`参数用于确定 NaN 值在排序序列中的位置。默认情况下，NaN 值存储在排序序列的最后。您可以将`na_position`参数设置为`“first”`来存储排序序列顶部的 NaN 值。
*   当我们对一个序列进行排序时，所有值的索引在排序时会被打乱。因此，排序序列中的索引是无序的。如果您想在排序后重置索引，您可以将`ignore_index`参数设置为 True。
*   `key`参数用于在排序前对系列执行操作。它接受一个矢量化函数作为其输入参数。提供给`key`参数的函数必须将熊猫系列作为其输入参数，并返回熊猫系列。排序前，该函数应用于系列。然后，函数输出中的值用于对序列进行排序。

## 在 Python 中对序列进行升序排序

要按升序对系列进行排序，可以对 series 对象使用`sort_values()` 方法，如下例所示。

```py
import pandas as pd
numbers=[12,34,11,25,27,8,13]
series=pd.Series(numbers)
print("The original series is:")
print(series)
sorted_series=series.sort_values()
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
0    12
1    34
2    11
3    25
4    27
5     8
6    13
dtype: int64
The sorted series is:
5     8
2    11
0    12
6    13
3    25
4    27
1    34
dtype: int64
```

在上面的例子中，我们首先创建了一个有 7 个数字的[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)。然后，我们使用`sort_values()`方法对序列进行排序。

您可以观察到，在对序列进行排序时，索引也随着序列中的值进行了洗牌。要重置索引，您可以将`ignore_index`参数设置为 True，如下所示。

```py
import pandas as pd
numbers=[12,34,11,25,27,8,13]
series=pd.Series(numbers)
print("The original series is:")
print(series)
sorted_series=series.sort_values(ignore_index=True)
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
0    12
1    34
2    11
3    25
4    27
5     8
6    13
dtype: int64
The sorted series is:
0     8
1    11
2    12
3    13
4    25
5    27
6    34
dtype: int64
```

在这个例子中，您可以观察到由`sort_values()` 方法返回的序列具有从 0 到 6 的索引，而不是混洗的索引。

## 按降序排列熊猫系列

要按降序对熊猫系列进行排序，可以将参数`sort_values()`中的参数`ascending`设置为 False。执行后，`sort_values()`方法将返回一个降序排列的序列。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
numbers=[12,34,11,25,27,8,13]
series=pd.Series(numbers)
print("The original series is:")
print(series)
sorted_series=series.sort_values(ascending=False,ignore_index=True)
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
0    12
1    34
2    11
3    25
4    27
5     8
6    13
dtype: int64
The sorted series is:
0    34
1    27
2    25
3    13
4    12
5    11
6     8
dtype: int64
```

在上面的例子中，我们已经将`sort_values()`方法中的`ascending`参数设置为 False。因此，在执行了`sort_values()`方法之后，我们得到了一个降序排列的序列。

## 在 Python 中对具有 NaN 值的序列进行排序

要用 NaN 值对 pandas 系列进行排序，只需调用 pandas 系列的 `sort_values()`方法，如下例所示。

```py
import pandas as pd
import numpy as np
numbers=[12,np.nan,11,np.nan,27,-8,13]
series=pd.Series(numbers)
print("The original series is:")
print(series)
series.sort_values(inplace=True,ignore_index=True)
print("The sorted series is:")
print(series)
```

输出:

```py
The original series is:
0    12.0
1     NaN
2    11.0
3     NaN
4    27.0
5    -8.0
6    13.0
dtype: float64
The sorted series is:
0    -8.0
1    11.0
2    12.0
3    13.0
4    27.0
5     NaN
6     NaN
dtype: float64
```

在此示例中，您可以观察到该系列包含 NaN 值。因此，默认情况下，`sort_values()`方法将 NaN 值放在排序序列的最后。如果您希望 NaN 值位于排序后的序列的开始处，您可以将`na_position`参数设置为`“first”` ，如下所示。

```py
import pandas as pd
import numpy as np
numbers=[12,np.nan,11,np.nan,27,-8,13]
series=pd.Series(numbers)
print("The original series is:")
print(series)
series.sort_values(inplace=True,ignore_index=True,na_position="first")
print("The sorted series is:")
print(series)
```

输出:

```py
The original series is:
0    12.0
1     NaN
2    11.0
3     NaN
4    27.0
5    -8.0
6    13.0
dtype: float64
The sorted series is:
0     NaN
1     NaN
2    -8.0
3    11.0
4    12.0
5    13.0
6    27.0
dtype: float64
```

在上面的两个例子中，您可以观察到序列的数据类型被设置为`float64`，不像前面的例子中序列的数据类型被设置为`int64`。这是因为在 python 中 NaN 值被认为是浮点[数据类型。因此，所有数字都被转换为最兼容的数据类型。](https://avidpython.com/python-basics/data-types-in-python/)

## 在 Python 中就地排序系列

在上面的例子中，您可以观察到原始序列没有被修改，我们得到了一个新的排序序列。如果您想对序列进行就地排序，您可以将`inplace`参数设置为 True，如下所示。

```py
import pandas as pd
numbers=[12,34,11,25,27,8,13]
series=pd.Series(numbers)
print("The original series is:")
print(series)
series.sort_values(inplace=True,ignore_index=True)
print("The sorted series is:")
print(series)
```

输出:

```py
The original series is:
0    12
1    34
2    11
3    25
4    27
5     8
6    13
dtype: int64
The sorted series is:
0     8
1    11
2    12
3    13
4    25
5    27
6    34
dtype: int64
```

在这个例子中，我们已经在`sort_values()`方法中将`inplace`参数设置为 True。因此，在执行了`sort_values()`方法之后，原始序列被排序，而不是创建一个新的熊猫序列。在这种情况下，`sort_values()`方法返回 None。

## 使用关键字对熊猫系列进行排序

默认情况下，序列中的值用于排序。现在，假设您想根据数值的大小而不是实际值对序列进行排序。为此，您可以使用 keys 参数。

我们将把`abs()` 函数传递给`sort_values()`方法的`key`参数。在此之后，序列的值将按其大小排序。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
numbers=[12,-34,11,-25,27,-8,13]
series=pd.Series(numbers)
print("The original series is:")
print(series)
series.sort_values(inplace=True,ignore_index=True,key=abs)
print("The sorted series is:")
print(series)
```

输出:

```py
The original series is:
0    12
1   -34
2    11
3   -25
4    27
5    -8
6    13
dtype: int64
The sorted series is:
0    -8
1    11
2    12
3    13
4   -25
5    27
6   -34
dtype: int64
```

在这个例子中，我们有一系列正数和负数。现在，为了使用数字的绝对值对熊猫系列进行排序，我们在 `sort_values()` 方法中使用了`key`参数。在`key`参数中，我们传递了`abs()`函数。

当执行`sort_values()`方法时，序列的元素首先被传递给`abs()` 函数。然后，由`abs()`函数返回的值被用来比较对序列排序的元素。这就是为什么我们得到的序列中的元素是按绝对值而不是实际值排序的。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能还会喜欢这篇关于用 Python](https://codinginfinite.com/regression-in-machine-learning-with-examples/) 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## Python 中的 sort_index()方法

除了使用值对序列进行排序，我们还可以使用行索引对序列进行排序。为此，我们可以使用`sort_index()`方法。它具有以下语法。

```py
Series.sort_index(*, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)
```

这里，

*   `axis`参数的使用方式与`sort_values()` 方法类似。
*   `level`参数用于当存在多级索引时，按某一级索引对序列进行排序。要按特定顺序按多个索引级别对序列进行排序，可以按相同顺序将级别列表传递给`level`参数。
*   默认情况下，series 对象按索引值升序排序。如果希望输出数据帧中的索引以降序排列，可以将`ascending`参数设置为 False。
*   执行后，`sort_values()` 方法返回排序后的序列。要通过索引对原始序列进行排序和修改，而不是创建一个新序列，可以将`inplace`参数设置为 True。
*   `kind`参数用于确定排序算法。默认情况下，使用`“quicksort”` 算法。如果索引值是一种特定的模式，在这种模式下另一种排序算法可能是有效的，那么您可以使用`‘mergesort’`、`‘heapsort’`或`‘stable’` 排序算法。
*   `na_position`参数用于确定 NaN 索引在排序序列中的位置。默认情况下，NaN 索引存储在排序序列的最后。您可以将`na_position`参数设置为 `“first”` 来存储排序序列顶部的 NaN 索引。
*   `sort_index()`方法按照特定的顺序(升序或降序)对索引进行排序。在对索引进行排序之后，如果您想要重置序列的索引，您可以将`ignore_index`参数设置为 True。
*   `key`参数用于在排序前对系列的索引执行操作。它接受一个矢量化函数作为其输入参数。提供给`key`参数的函数必须将索引作为其输入参数，并返回一个熊猫系列。在排序之前，该函数应用于索引。然后，函数输出中的值用于对序列进行排序。

## 按索引升序排列熊猫系列

要按索引升序对 pandas 系列进行排序，可以调用 series 对象上的`sort_index()`方法，如下例所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
sorted_series=series.sort_index()
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
3        a
23       b
11       c
14      ab
16     abc
2     abcd
45      bc
65       d
dtype: object
The sorted series is:
2     abcd
3        a
11       c
14      ab
16     abc
23       b
45      bc
65       d
dtype: object
```

在这个例子中，我们有一系列以数字为索引的字符串。由于我们已经对 pandas 系列使用了`sort_index()`方法进行排序，所以该系列是按索引值排序的。因此，我们得到了一个对索引值进行排序的序列。

排序后，如果要重置输出数据帧的索引，可以在如下所示的`sort_index()`方法中将`ignore_index`参数设置为 True。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
sorted_series=series.sort_index(ignore_index=True)
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
3        a
23       b
11       c
14      ab
16     abc
2     abcd
45      bc
65       d
dtype: object
The sorted series is:
0    abcd
1       a
2       c
3      ab
4     abc
5       b
6      bc
7       d
dtype: object
```

在这个例子中，我们已经在`sort_index()`方法中将`ignore_index`参数设置为 True。因此，在按原始索引值对序列进行排序后，序列的索引将被重置。

## 在 Python 中按索引降序对序列进行排序

要按索引降序对 pandas 系列进行排序，可以将`sort_index()`方法中的`ascending`参数设置为 False，如下例所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
sorted_series=series.sort_index(ascending=False)
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
3        a
23       b
11       c
14      ab
16     abc
2     abcd
45      bc
65       d
dtype: object
The sorted series is:
65       d
45      bc
23       b
16     abc
14      ab
11       c
3        a
2     abcd
dtype: object
```

在这个例子中，我们将`sort_index()`方法中的升序参数设置为 False。因此，该系列按索引降序排序。

## 按具有 NaN 值的索引对熊猫系列进行排序

要在索引中有 NaN 值时按索引对序列进行排序，只需调用 pandas 序列上的`sort_index()`方法，如下例所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
sorted_series=series.sort_index()
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
3.0        a
23.0       b
NaN        c
14.0      ab
16.0     abc
NaN     abcd
45.0      bc
65.0       d
dtype: object
The sorted series is:
3.0        a
14.0      ab
16.0     abc
23.0       b
45.0      bc
65.0       d
NaN        c
NaN     abcd
dtype: object
```

在上面的示例中，序列的索引包含 NaN 值。默认情况下，NaN 值存储在排序序列的最后。如果您希望 NaN 值位于排序序列的开始处，您可以将`na_position`参数设置为 `“first”`，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
sorted_series=series.sort_index(na_position="first")
print("The sorted series is:")
print(sorted_series)
```

输出:

```py
The original series is:
3.0        a
23.0       b
NaN        c
14.0      ab
16.0     abc
NaN     abcd
45.0      bc
65.0       d
dtype: object
The sorted series is:
NaN        c
NaN     abcd
3.0        a
14.0      ab
16.0     abc
23.0       b
45.0      bc
65.0       d
dtype: object
```

在这个例子中，您可以看到我们已经在`sort_index()` 方法中将`na_position`参数设置为`"first"`。因此，将 NaN 值作为索引的元素被保存在由`sort_index()`方法返回的排序序列的开始处。

有趣阅读:[做程序员的优势](https://www.codeconquest.com/blog/advantages-of-being-a-programmer-in-2022/)。

## 在 Python 中按索引就地对序列排序

默认情况下，`sort_index()`方法不会对原始序列进行排序。它返回一个按索引排序的新系列。如果您想要修改原始序列，您可以在如下所示的`sort_index()`方法中将`inplace`参数设置为 True。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,np.nan,14,16,np.nan,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series.sort_index(inplace=True)
print("The sorted series is:")
print(series)
```

输出:

```py
The original series is:
3.0        a
23.0       b
NaN        c
14.0      ab
16.0     abc
NaN     abcd
45.0      bc
65.0       d
dtype: object
The sorted series is:
3.0        a
14.0      ab
16.0     abc
23.0       b
45.0      bc
65.0       d
NaN        c
NaN     abcd
dtype: object
```

在这个例子中，我们已经在`sort_index()`方法中将`inplace`参数设置为 True。因此，原始系列被排序，而不是创建新系列。

## 使用 Python 中的键按索引对熊猫系列进行排序

通过使用`key`参数，我们可以在按索引对序列排序之前对序列的索引执行操作。例如，如果序列中的索引是负数，并且希望使用索引的大小对序列进行排序，可以将`abs()`函数传递给`sort_index()`方法中的`key`参数。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,-100,14,16,-3,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series.sort_index(inplace=True,key=abs)
print("The sorted series is:")
print(series)
```

输出:

```py
The original series is:
 3         a
 23        b
-100       c
 14       ab
 16      abc
-3      abcd
 45       bc
 65        d
dtype: object
The sorted series is:
 3         a
-3      abcd
 14       ab
 16      abc
 23        b
 45       bc
 65        d
-100       c
dtype: object
```

在这个例子中，我们有一个以正数和负数作为索引的序列。现在，为了使用指数的绝对值对熊猫系列进行排序，我们使用了`sort_index()`方法中的关键参数。在`key`参数中，我们传递了`abs()` 函数。

当执行`sort_index()`方法时，序列的索引首先被传递给`abs()`函数。由`abs()`函数返回的值然后被用来比较排序序列的索引。这就是为什么我们得到的序列中的指数是按绝对值而不是实际值排序的。

## 结论

在本文中，我们讨论了如何用 Python 对熊猫系列进行排序。为此，我们使用了`sort_values()`和`sort_index()`方法。我们使用这些方法的不同参数演示了不同的示例。

我希望你喜欢阅读这篇文章。要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

快乐学习！