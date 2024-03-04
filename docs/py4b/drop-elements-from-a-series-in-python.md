# 在 Python 中从系列中删除元素

> 原文：<https://www.pythonforbeginners.com/basics/drop-elements-from-a-series-in-python>

Pandas 系列对于处理具有有序键值对的数据非常有用。在本文中，我们将讨论从熊猫系列中删除元素的不同方法。

## 使用 Drop()方法从熊猫系列中删除元素

我们可以使用`drop()`方法从[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)中删除元素。它具有以下语法。

```py
Series.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
```

这里，

*   `labels`参数获取我们需要从序列中删除的元素的索引。您可以将单个索引标签或一个索引列表传递给`labels`参数。
*   `axis`参数用于决定我们是否要删除一行或一列。对于熊猫系列，不使用`axis`参数。在函数中定义它只是为了确保`drop()` 方法与[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)的兼容性。
*   `index`参数用于为数据帧中的给定标签选择要删除的元素的索引。`index`参数对于系列对象是多余的。但是，您可以使用`index`参数代替`labels`参数。
*   `columns`参数用于选择数据帧中要删除的列。`“columns”` 参数在这里也是多余的。您可以使用`labels`或`index`参数从系列中删除元素。
*   当序列包含多索引时，`levels`参数用于从序列中删除元素。`levels`参数获取需要为指定标签删除元素的级别或级别列表。
*   默认情况下，`drop()`方法在从原始序列中删除元素后返回一个新的序列对象。在此过程中，原始系列不会被修改。如果您想要修改原始系列而不是创建新系列，您可以将`inplace`参数设置为 True。
*   `drop()`方法在从序列中删除元素时遇到错误时会引发异常。例如，如果我们想要删除的索引或标签在序列中不存在，那么`drop()`方法会引发一个 python KeyError 异常。要在从序列中删除元素时抑制此类错误，您可以将 errors 参数设置为`“ignore”`。

执行后，如果`inplace`参数设置为 False，`drop()`方法将返回一个新的序列。否则，它返回 None。

## 从熊猫系列中删除一个元素

要从序列中删除单个元素，可以将元素的索引传递给 `drop()`方法中的`labels`参数，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series=series.drop(labels=11)
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
3        a
23       b
14      ab
16     abc
2     abcd
45      bc
65       d
dtype: object
```

在上面的例子中，我们首先使用`Series()`构造函数创建了一个 Series 对象。然后我们使用`drop()`方法删除索引为 11 的元素。为此，我们将值 11 传递给了`drop()`方法。在执行了`drop()`方法之后，您可以观察到索引为 11 的元素已经从输出序列中删除了。

除了`labels`参数，您还可以在如下所示的`drop()`方法中使用`index`参数。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series=series.drop(index=11)
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
3        a
23       b
14      ab
16     abc
2     abcd
45      bc
65       d
dtype: object
```

在本例中，我们使用了`index`参数，而不是`labels`参数。然而，在两种情况下，执行`drop()`方法后的结果序列是相同的。

## 从熊猫系列中删除多个元素

要从一个序列中删除多个元素，可以向参数`labels`传递一个要删除的元素索引的 [python 列表](https://www.pythonforbeginners.com/lists/python-lists-cheat-sheet-2)。例如，如果您想要删除给定序列中索引 11、16 和 2 处的元素，您可以将列表`[11,16,2]`传递给`drop()`方法中的`labels`参数，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series=series.drop(labels=[11,16,2])
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
3      a
23     b
14    ab
45    bc
65     d
dtype: object
```

在这个例子中，我们将列表`[11, 16, 2]`作为输入传递给了参数`labels`。因此，在执行了`drop()`方法之后，索引 11、16 和 2 处的元素将从原来的 series 对象中删除。

您可以将索引列表传递给`index`参数，而不是传递给`labels`参数，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series=series.drop(index=[11,16,2])
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
3      a
23     b
14    ab
45    bc
65     d
dtype: object
```

## 将熊猫系列中的元素放到适当的位置

默认情况下，`drop()`方法返回一个新的序列，并且不会从原始序列中删除指定的元素。要从 pandas 系列中删除元素，可以将参数`inplace`设置为 True，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series.drop(index=[11,16,2],inplace=True)
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
3      a
23     b
14    ab
45    bc
65     d
dtype: object
```

在所有前面的例子中，`drop()`方法返回一个新的 Series 对象。在这个例子中，我们已经在`drop()`方法中将`inplace`参数设置为 True。因此，从原始系列中删除元素，并对其进行修改。在这种情况下， `drop()`方法返回 None。

## 如果索引存在，则从序列中删除元素

当使用`drop()`方法从序列中删除元素时，我们可能会向标签或索引参数传递一个不在序列对象中的索引。如果传递给标签或索引参数的值在序列中不存在，`drop()` 方法会遇到一个 [KeyError 异常](https://www.pythonforbeginners.com/basics/python-keyerror)，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series.drop(index=1117,inplace=True)
print("The modified series is:")
print(series)
```

输出:

```py
KeyError: '[1117] not found in axis'
```

在上面的例子中，我们已经将值`1117`传递给了参数`index`。由于值 1117 在序列中不存在，我们得到一个 KeyError 异常。

如果索引存在，为了避免错误并删除序列中的元素，可以使用`errors`参数。默认情况下，`errors`参数设置为`"raise"`。因此，`drop()` 方法每次遇到错误时都会引发一个异常。为了[抑制异常](https://www.pythonforbeginners.com/basics/suppress-exceptions-in-python)，可以将错误参数设置为`“ignore”` ，如下例所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The original series is:")
print(series)
series.drop(index=1117,inplace=True,errors="ignore")
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
3        a
23       b
11       c
14      ab
16     abc
2     abcd
45      bc
65       d
dtype: object
```

在上面的例子中，我们已经将值`1117`传递给了 index 参数。因为 1117 不在序列索引中，所以`drop()`方法会遇到一个 KeyError 异常。然而，我们已经在`drop()`方法中将`errors`参数设置为 `"ignore"`。因此，它抑制了误差。您还可以观察到，`drop()`方法返回的序列与原始序列相同。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能还会喜欢这篇关于用 Python](https://codinginfinite.com/regression-in-machine-learning-with-examples/) 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## 从熊猫系列中删除 NaN 值

NaN 值是 Python 中具有浮点[数据类型的特殊数字。NaN 值用于表示没有值。大多数情况下，NaN 值在给定的数据集中并不重要，我们需要删除这些值。](https://avidpython.com/python-basics/data-types-in-python/)

您可以使用`dropna()`方法从 pandas 系列中删除 NaN 值。它具有以下语法。

```py
Series.dropna(*, axis=0, inplace=False, how=None)
```

这里，

*   `axis`参数用于决定我们是否要从序列的行或列中删除 nan 值。对于熊猫系列，不使用`axis`参数。定义它只是为了确保`dropna()` 方法与 pandas 数据帧的兼容性。
*   默认情况下， `dropna()`方法在从原始系列中删除 nan 值后返回一个新的系列对象。在此过程中，原始系列不会被修改。如果您想从原始序列中删除 nan 值，而不是创建一个新序列，您可以将`inplace`参数设置为 True。
*   `“how”`参数不用于系列。

执行后，如果`inplace`参数设置为 False，`dropna()`方法将返回一个新的序列。否则，它返回 None。

您可以从 pandas 系列中删除 nan 值，如下例所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c",np.nan,"ab","abc",np.nan,"abcd","bc","d"]
series=pd.Series(letters)
print("The original series is:")
print(series)
series=series.dropna()
print("The modified series is:")
print(series)
```

输出:

```py
The original series is:
0       a
1       b
2       c
3     NaN
4      ab
5     abc
6     NaN
7    abcd
8      bc
9       d
dtype: object
The modified series is:
0       a
1       b
2       c
4      ab
5     abc
7    abcd
8      bc
9       d
```

在上面的示例中，您可以观察到原始系列有两个 NaN 值。执行后，`dropna()` 方法删除 NaN 值及其索引，并返回一个新的序列。

## 从熊猫系列中删除 NaN 值

如果您想从原始序列中删除 NaN 值，而不是创建一个新序列，您可以在如下所示的`dropna()`方法中将`inplace`参数设置为 True。

```py
import pandas as pd
import numpy as np
letters=["a","b","c",np.nan,"ab","abc",np.nan,"abcd","bc","d"]
series=pd.Series(letters)
print("The original series is:")
print(series)
series.dropna(inplace=True)
print("The modified series is:")
print(series)
```

输出:

```py
import pandas as pd
import numpy as np
letters=["a","b","c",np.nan,"ab","abc",np.nan,"abcd","bc","d"]
series=pd.Series(letters)
print("The original series is:")
print(series)
series.dropna(inplace=True)
print("The modified series is:")
print(series)
```

这里，我们已经将`inplace`参数设置为 True。因此，`dropna()`方法修改了原始序列，而不是创建一个新序列。在这种情况下，`dropna()`方法在执行后返回 None。

## 从熊猫系列中删除副本

我们[数据预处理](https://codinginfinite.com/data-preprocessing-explained/)，我们经常需要从给定的数据中删除重复值。要删除 pandas 系列中的重复值，可以使用`drop_duplicates()` 方法。它具有以下语法。

```py
Series.drop_duplicates(*, keep='first', inplace=False)
```

这里，

*   `keep`参数用于决定在删除重复项后我们需要保留哪些值。要删除除第一次出现之外的所有重复值，可以将参数`keep`设置为默认值`“first”` 。要删除除最后一次出现之外的所有重复值，您可以将`keep`参数设置为`“last”`。如果想丢弃所有重复值，可以将`keep`参数设置为 False。
*   默认情况下，`drop_duplicates()`方法在从原始序列中删除重复值后返回一个新的序列对象。在此过程中，原始系列不会被修改。如果您想删除原始序列中的重复值，而不是创建一个新序列，您可以将`inplace`参数设置为 True。

执行后，如果`inplace`参数设置为 False，`drop_duplicates()`方法将返回一个新的序列。否则，它返回 None。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
letters=["a","b","a","a","ab","abc","ab","abcd","bc","abc","ab"]
series=pd.Series(letters)
print("The original series is:")
print(series)
series=series.drop_duplicates()
print("The modified series is:")
print(series)
```

输出:

```py
The original series is:
0        a
1        b
2        a
3        a
4       ab
5      abc
6       ab
7     abcd
8       bc
9      abc
10      ab
dtype: object
The modified series is:
0       a
1       b
4      ab
5     abc
7    abcd
8      bc
dtype: object
```

在上面的示例中，您可以观察到字符串“a”、“ab”和“abc”在序列中出现了多次。因此，当我们在 series 对象上调用`drop_duplicates()` 方法时，除了字符串的一次出现之外，所有的重复都将从 series 中删除。

查看索引，您可以看到，如果元素在序列中出现多次，则保留了元素的第一次出现。如果希望保留最后出现的具有重复值的元素，可以将`keep`参数设置为 `"last"`，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","a","a","ab","abc","ab","abcd","bc","abc","ab"]
series=pd.Series(letters)
print("The original series is:")
print(series)
series=series.drop_duplicates(keep="last")
print("The modified series is:")
print(series)
```

输出:

```py
The original series is:
0        a
1        b
2        a
3        a
4       ab
5      abc
6       ab
7     abcd
8       bc
9      abc
10      ab
dtype: object
The modified series is:
1        b
3        a
7     abcd
8       bc
9      abc
10      ab
dtype: object
```

在上面的例子中，我们已经将 keep 参数设置为`"last"`。因此，您可以观察到`drop_duplicates()` 方法保留了具有重复值的元素的最后一次出现。

## 在熊猫系列中放置副本

默认情况下，`drop_duplicates()` 方法不会修改原始的 series 对象。它返回一个新系列。如果您想通过删除重复序列来修改原始序列，您可以在如下所示的`drop_duplicates()` 方法中将`inplace`参数设置为 True。

```py
import pandas as pd
import numpy as np
letters=["a","b","a","a","ab","abc","ab","abcd","bc","abc","ab"]
series=pd.Series(letters)
print("The original series is:")
print(series)
series.drop_duplicates(inplace=True)
print("The modified series is:")
print(series)
```

输出:

```py
The original series is:
0        a
1        b
2        a
3        a
4       ab
5      abc
6       ab
7     abcd
8       bc
9      abc
10      ab
dtype: object
The modified series is:
0       a
1       b
4      ab
5     abc
7    abcd
8      bc
dtype: object
```

在这个例子中，我们已经将`inplace`参数设置为真。因此，`drop_duplicates()` 方法修改了原始序列，而不是创建一个新序列。在这种情况下，d `rop_duplicates()`方法在执行后返回 None。

## 删除熊猫系列中的所有重复值

要删除熊猫系列中的所有副本，可以将参数`keep`设置为 False，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","a","a","ab","abc","ab","abcd","bc","abc","ab"]
series=pd.Series(letters)
print("The original series is:")
print(series)
series=series.drop_duplicates(keep=False)
print("The modified series is:")
print(series)
```

输出:

```py
The original series is:
0        a
1        b
2        a
3        a
4       ab
5      abc
6       ab
7     abcd
8       bc
9      abc
10      ab
dtype: object
The modified series is:
1       b
7    abcd
8      bc
dtype: object
```

在这个例子中，我们已经在 `drop_duplicates()`方法中将`keep`参数设置为 False。因此，您可以看到所有具有重复值的元素都从序列中删除了。

## 结论

在本文中，我们讨论了从熊猫系列中删除元素的不同方法。要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)