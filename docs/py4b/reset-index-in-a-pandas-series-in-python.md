# 在 Python 中重置熊猫系列的索引

> 原文：<https://www.pythonforbeginners.com/basics/reset-index-in-a-pandas-series-in-python>

Pandas 系列对象在 python 中用于处理顺序数据。为了处理一个序列中的数据，我们通常使用元素的索引。在本文中，我们将讨论如何在熊猫系列中建立索引。

## 如何重置熊猫系列的索引？

为了重置[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)的索引，我们可以使用两种方法。首先，我们可以将包含新索引的列表直接分配给 series 对象的`index`属性。或者，我们可以使用`reset_index()` 方法。让我们讨论这两种方法。

## 使用 Index 属性重置序列中的索引

`index`属性存储熊猫系列的[索引。为了重置一个系列对象的索引，我们将首先使用`len()`函数找到系列对象的长度。`len()`函数将一个 iterable 对象作为其输入参数，并返回长度。](https://www.pythonforbeginners.com/basics/rename-index-in-a-pandas-series)

找到序列的长度后，我们将使用`range()`函数创建一个 range 对象，包含从 0 到序列长度的数字。`range()`函数将范围内的最大值作为其输入参数，并返回一个从 0 到(最大值-1)的范围对象。最后，我们将把 range 对象分配给系列的`index`属性。在执行上述语句后，我们将得到一系列新的指数。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
lenSeries=len(series)
indices=range(lenSeries)
series.index=indices
print("The modified series is:")
print(series)
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
The modified series is:
0       a
1       b
2       c
3      ab
4     abc
5    abcd
6      bc
7       d
dtype: object
```

在上面的例子中，您可以观察到原始数据帧中的索引已经被移除，并且从 0 到 7 的新索引已经被分配给元素。

在上述方法中，您需要确保分配给序列的索引属性的 python 列表的长度必须等于序列中元素的数量。否则，程序将会遇到 [ValueError](https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10) 异常。

我们可以使用`reset_index()`方法来重置熊猫系列的索引，而不是使用`index`属性。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇 [MLFlow 教程，里面有代码示例](https://codinginfinite.com/mlflow-tutorial-with-code-example/)。您可能还会喜欢这篇关于用 Python 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## reset index()方法

`reset_index()`方法的语法如下。

```py
Series.reset_index(level=None, *, drop=False, name=_NoDefault.no_default, inplace=False, allow_duplicates=False)
```

这里，

*   `level`参数用于选择多级索引时需要删除的索引级别。您可以将需要从序列中删除的索引的级别、级别列表、索引名称或索引名称列表传递给`level`参数。
*   默认情况下，当我们使用`reset_index()`方法重置一个系列的索引时，该索引被添加为一个额外的列，并且我们从`reset_index()`方法得到一个 dataframe 作为输出，而不是一个系列。如果您想删除索引而不是将其转换为列，可以将`drop`参数设置为 True。
*   如果您想在`reset_index()`方法的输出中使用原始序列的索引作为新列，您可以使用 name 参数来设置包含数据值的列的名称。默认情况下，包含索引值的列的名称被设置为`“index”` 。在`drop`参数设置为真的情况下，`name`参数将被忽略。
*   默认情况下，`reset_index()`方法在重置索引后返回一个新的序列。要修改原始序列，您可以将`inplace`参数设置为`True`。
*   `allow_duplicates`参数用于决定系列中是否允许重复的列标签。要重置系列的索引，`allow_duplicates`参数没有用。

## 使用 reset_index()方法重置序列中的索引

要重置序列中的索引，只需调用 series 对象上的`reset_index()`方法，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series=series.reset_index()
print("The modified series is:")
print(series)
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
The modified series is:
   index     0
0      3     a
1     23     b
2     11     c
3     14    ab
4     16   abc
5      2  abcd
6     45    bc
7     65     d
```

您可以观察到`reset_index()` 方法返回了一个 dataframe。`reset_index()` 方法将当前索引提升为一列，并返回一个[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)而不是一个序列。

这里，包含数据值的列的名称被设置为 0。您可以使用`reset_index()`方法中的 name 参数设置数据列的名称，如下例所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series=series.reset_index(name="letters")
print("The modified series is:")
print(series)
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
The modified series is:
   index letters
0      3       a
1     23       b
2     11       c
3     14      ab
4     16     abc
5      2    abcd
6     45      bc
7     65       d
```

在上面的例子中，我们将文字`"letters"`传递给了`reset_index()`方法中的 name 参数。因此，当执行`reset_index()`方法时，它返回一个包含两列的数据帧，即 index 和 letters。这里，`"index"`是根据序列的原始索引创建的列的名称。然而，`"letters"`是序列中包含数据的列的名称。

如果您不想在重置索引时创建 dataframe 并删除索引列，您可以将`reset_index()`方法中的`drop`参数设置为 True，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series=series.reset_index(drop=True)
print("The modified series is:")
print(series)
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
The modified series is:
0       a
1       b
2       c
3      ab
4     abc
5    abcd
6      bc
7       d
dtype: object
```

在上面的例子中，我们已经将`drop`参数设置为 True。因此，`reset_index()`方法返回一个序列而不是一个数据帧。

## 使用 reset_index()方法就地重置索引

默认情况下，`rest_index()` 方法返回一个新的序列。如果要重置原始序列中的索引，可以将`inplace`参数设置为 True，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series.reset_index(drop=True,inplace=True)
print("The modified series is:")
print(series)
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
The modified series is:
0       a
1       b
2       c
3      ab
4     abc
5    abcd
6      bc
7       d
dtype: object
```

在这个例子中，我们已经在`reset_index()` 方法中将`inplace`参数设置为 True。因此，这些指数将从原始序列中删除，而不是创建一个新的序列。在这种情况下，`reset_index()`方法在执行后返回 None。

## 结论

在本文中，我们讨论了在熊猫系列中重置索引的不同方法。要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！