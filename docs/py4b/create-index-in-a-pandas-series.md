# 在熊猫系列中创建索引

> 原文：<https://www.pythonforbeginners.com/basics/create-index-in-a-pandas-series>

Pandas 系列对象用于存储数据，当我们需要使用它的位置和标签来访问它时。在本文中，我们将讨论在熊猫系列中创建索引的不同方法。

## 使用 Index 参数在熊猫系列中创建索引

当我们创建一个[熊猫系列](https://www.pythonforbeginners.com/basics/pandas-series-data-structure-in-python)时，它有一个默认索引，从 0 开始到系列长度-1。例如，考虑下面的例子。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
series=pd.Series(letters)
print("The series is:")
print(series)
```

输出:

```py
The series is:
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

在上面的例子中，我们创建了一系列的 8 个元素。您可以观察到序列中元素的索引从 0 到 7 进行编号。这些是默认索引。

如果您想为序列分配一个自定义索引，您可以使用`Series()`构造函数中的`index`参数。`Series()`构造函数中的 index 参数获取一个元素数量与序列中的元素数量相等的列表，并为序列创建一个自定义索引，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters,index=numbers)
print("The series is:")
print(series)
```

输出:

```py
The series is:
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

在上面的例子中，我们已经将 [python 列表](https://avidpython.com/python-basics/python-list/) [3，23，11，14，16，2，45，65]传递给 Series()构造函数的 index 参数。在执行了`Series()`构造函数之后，这个列表的元素被指定为序列中元素的索引。

## 使用 Index 属性在熊猫系列中创建索引

您也可以在创建系列后为系列创建新的索引。例如，如果您想将其他值指定为序列中的索引，您可以使用 series 对象的`index`属性。要创建一个新的定制索引，您可以为属性`index`分配一个值列表，如下所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The series is:")
print(series)
```

输出:

```py
The series is:
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

在本例中，我们在创建系列后，将列表`[3, 23, 11, 14, 16, 2, 45, 65]`分配给系列的`index`属性。因此，该列表的元素被指定为序列中元素的索引。

这里，传递给 index 属性的列表的长度必须等于序列中元素的数量。否则，程序会遇到[值错误异常](https://www.pythonforbeginners.com/exceptions/valueerror-invalid-literal-for-int-with-base-10)。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65,117]
series=pd.Series(letters)
series.index=numbers
print("The series is:")
print(series)
```

输出:

```py
ValueError: Length mismatch: Expected axis has 8 elements, new values have 9 elements
```

在上面的例子中，您可以观察到列表`"letters"` 只有 8 个元素。因此，该系列仅包含 8 个元素。另一方面，列表`"numbers"` 具有 9 个元素。因此，当我们将`"numbers"` 列表赋给序列的索引属性时，程序会遇到 ValueError 异常。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇 [MLFlow 教程，里面有代码示例](https://codinginfinite.com/mlflow-tutorial-with-code-example/)。您可能还会喜欢这篇关于用 Python 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## 使用 set_axis()方法在熊猫系列中创建一个索引

我们可以使用`set_axis()` 方法在熊猫系列中创建一个索引，而不是使用`index`属性。

### set_axis()方法

set_axis()方法具有以下语法。

```py
Series.set_axis(labels, *, axis=0, inplace=_NoDefault.no_default, copy=_NoDefault.no_default)
```

这里，

*   `labels`参数接受一个包含索引值的类似列表的对象。您还可以将一个索引对象传递给`labels`参数。传递给 labels 参数的任何对象中的元素数量应该与调用`set_axis()`方法的序列中的元素数量相同。
*   `axis`参数用于决定我们是否想要为行或列创建索引。因为一个系列只有一列，所以没有使用`axis`参数。
*   创建新索引后，`set_axis()`方法返回一个新的 Series 对象。如果要修改原来的系列对象，可以将`inplace`参数设置为 True。
*   `copy`参数用于决定是否复制底层数据，而不是修改原始序列。默认情况下，这是真的。

为了使用`set_axis()`方法创建一个索引，我们将在原始的 series 对象上调用这个方法。我们将把包含新索引值的列表作为输入参数传递给`set_axis()`方法。执行后，`set_axis()`方法将返回一个新的序列，该序列的索引已被修改。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series=series.set_axis(labels=numbers)
print("The series is:")
print(series)
```

输出:

```py
The series is:
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

在这个例子中，我们首先创建了一个包含 8 个元素的序列。然后，我们使用`set_index()`方法为序列中的元素分配新的索引。您可以观察到`set_index()` 方法返回了一个新的序列。因此，原始系列不会被修改。若要通过分配新的索引而不是创建新的索引来修改原始序列，可以在序列中就地创建索引。

## 在熊猫系列中就地创建索引

要在 pandas 系列中就地创建索引，可以将新索引分配给 series 对象的`index`属性，如下例所示。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.index=numbers
print("The series is:")
print(series)
```

输出:

```py
The series is:
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

您也可以使用`set_axis()` 方法在序列中创建一个索引。为此，您可以将包含新索引值的列表传递给`set_axis()` 方法，并在调用原始 series 对象上的`set_axis()`方法时将`inplace`参数设置为 True。执行后，您将获得如下所示的修改后的 series 对象。

```py
import pandas as pd
import numpy as np
letters=["a","b","c","ab","abc","abcd","bc","d"]
numbers=[3,23,11,14,16,2,45,65]
series=pd.Series(letters)
series.set_axis(labels=numbers,inplace=True)
print("The series is:")
print(series)
```

输出:

```py
The series is:
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

在这个例子中，我们使用了`set_index()` 方法为序列中的元素分配新的索引。您可以观察到我们已经在`set_index()` 方法中将`inplace`参数设置为 True。因此，新的索引是在原始系列对象本身中分配的。

使用`inplace`参数时，您将得到一个表示`"FutureWarning: Series.set_axis 'inplace' keyword is deprecated and will be removed in a future version. Use obj = obj.set_axis(..., copy=False) instead"`的未来警告。这意味着`inplace`参数已经被弃用。因此，如果在熊猫的未来版本中使用相同的代码，程序可能会出错。为了避免这种情况，您可以使用`copy`参数。

默认情况下，`copy`参数设置为真。因此，`set_axis()`方法使用原始序列的副本并修改它。如果您想修改原始序列，您可以在`set_axis()`方法中将`copy`参数设置为 False。

## 结论

在本文中，我们讨论了用 Python 在 pandas 系列中创建索引的不同方法。要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)