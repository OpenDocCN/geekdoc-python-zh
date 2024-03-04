# 检查熊猫数据框架中的列是否排序

> 原文：<https://www.pythonforbeginners.com/basics/check-if-a-column-is-sorted-in-a-pandas-dataframe>

Pandas dataframe 是用 python 处理表格数据的一个很好的工具。在本文中，我们将讨论不同的方法来检查一个列是否在 pandas 数据帧中排序。

## 检查是否使用列属性对列进行排序

为了检查一个列是否在[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)中按升序排序，我们可以使用该列的`is_monotonic`属性。如果列按升序排序，即如果列中的值单调递增，则`is_monotonic`属性的计算结果为`True`。

例如，如果数据帧按升序排序，`is_monotonic`属性将计算为 True，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
df.sort_values(by="Marks",inplace=True)
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
7      3    11       Bobby     50     D
0      2    27       Harsh     55     C
5      3    27      Aditya     55     C
1      2    23       Clara     78     B
4      3    15    Prashant     78     B
6      3    23  Radheshyam     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
The 'Marks' column is sorted: True
```

在上面的例子中，我们首先使用`read_csv()`函数将一个 CSV 文件加载到一个数据帧中。之后，我们[使用`sort_values()` 方法按照`"Marks"`列对数据帧](https://www.pythonforbeginners.com/basics/sort-pandas-dataframe-in-python)进行排序。排序后，可以观察到该列的`is_monotonic`属性返回 True。它表示该列按降序排序。

如果一列按降序排序，`is_monotonic`属性的计算结果将为 False。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
df.sort_values(by="Marks",inplace=True,ascending=False)
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
3      3    34         Amy     88     A
2      3    33        Tina     82     A
1      2    23       Clara     78     B
4      3    15    Prashant     78     B
6      3    23  Radheshyam     78     B
0      2    27       Harsh     55     C
5      3    27      Aditya     55     C
7      3    11       Bobby     50     D
The 'Marks' column is sorted: False
```

在本例中，我们已经按降序对`"Marks"`列进行了排序。因此，`is_monotonic`属性的计算结果为 False。

如果 dataframe 中的某一列没有排序，`is_monotonic`属性的计算结果将为 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
The 'Marks' column is sorted: False
```

在这里，您可以观察到我们已经访问了 is_monotonic 属性，而没有按照`"Marks"` 列对 dataframe 进行排序。因此，`"Marks"` 列是未排序的，并且`is_monotonic`属性的计算结果为 False。

`is_monotonic`不支持`NaN`值。如果一列包含`NaN`值，那么`is_monotonic`属性的计算结果总是假。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade.csv")
df.sort_values(by="Marks",inplace=True,ascending=True)
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
6       2    27       Harsh   55.0     C
10      3    27      Aditya   55.0     C
4       2    22         Tom   73.0     B
2       1    14         Sam   75.0     B
5       2    15        Golu   79.0     B
0       1    11      Aditya   85.0     A
8       3    34         Amy   88.0     A
1       1    12       Chris    NaN     A
3       1    15       Harry    NaN   NaN
7       2    23       Clara    NaN     B
9       3    15    Prashant    NaN     B
11      3    23  Radheshyam    NaN   NaN
The 'Marks' column is sorted: False
```

在这个例子中，您可以观察到`"Marks"`列包含了`NaN`值。因此，即使在排序之后,`is_monotonic`属性的计算结果也是假的。您可能会认为 NaN 值在列的最后。也许，这就是为什么`is_monotonic`属性评估为 False。

但是，如果我们将具有 NaN 值的行放在 dataframe 的顶部，`is_monotonic`属性将再次计算为 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade.csv")
df.sort_values(by="Marks",inplace=True,ascending=True,na_position="first")
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
1       1    12       Chris    NaN     A
3       1    15       Harry    NaN   NaN
7       2    23       Clara    NaN     B
9       3    15    Prashant    NaN     B
11      3    23  Radheshyam    NaN   NaN
6       2    27       Harsh   55.0     C
10      3    27      Aditya   55.0     C
4       2    22         Tom   73.0     B
2       1    14         Sam   75.0     B
5       2    15        Golu   79.0     B
0       1    11      Aditya   85.0     A
8       3    34         Amy   88.0     A
The 'Marks' column is sorted: False
```

在本例中，我们将 NaN 值放在排序后的`"Marks"`列的开头。即使在这之后，`is_monotonic`属性的计算结果也是 False。因此，我们可以得出结论，`is_monotonic`属性不能用于具有 NaN 值的列。

在使用`is_monotonic`属性时，您将得到一个`FutureWarning`,并显示消息**“future warning:is _ monotonic 已被否决，并将在未来版本中被删除。请改用 is_monotonic_increasing。因此，在未来的熊猫版本中，`is_monotonic`属性将被弃用。作为替代，我们可以使用`is_monotonic_increasing`和`is_monotonic_decreasing`属性来检查熊猫数据帧中的列是否排序。**

### 检查数据帧中的列是否按升序排序

要检查数据帧中的列是否按升序排序，我们可以使用`is_monotonic_increasing`属性。如果一个列按升序排序，那么`is_monotonic_increasing`属性的计算结果为 True。否则，它被设置为 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
df.sort_values(by="Marks",inplace=True,ascending=True)
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_increasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
7      3    11       Bobby     50     D
0      2    27       Harsh     55     C
5      3    27      Aditya     55     C
1      2    23       Clara     78     B
4      3    15    Prashant     78     B
6      3    23  Radheshyam     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
The 'Marks' column is sorted: True
```

如果一列没有排序，`is_monotonic_increasing`属性的计算结果为 False。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_increasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
The 'Marks' column is sorted: False
```

此外，如果一列按降序排序，`is_monotonic_increasing`属性的计算结果为 False。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
df.sort_values(by="Marks",inplace=True,ascending=False)
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_increasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
3      3    34         Amy     88     A
2      3    33        Tina     82     A
1      2    23       Clara     78     B
4      3    15    Prashant     78     B
6      3    23  Radheshyam     78     B
0      2    27       Harsh     55     C
5      3    27      Aditya     55     C
7      3    11       Bobby     50     D
The 'Marks' column is sorted: False
```

`is_monotonic_increasing`属性不能用于具有 NaN 值的列。如果一个列有 NaN 值，那么`is_monotonic_increasing`属性的计算结果总是 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade.csv")
df.sort_values(by="Marks",inplace=True,ascending=True,na_position="last")
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_increasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
6       2    27       Harsh   55.0     C
10      3    27      Aditya   55.0     C
4       2    22         Tom   73.0     B
2       1    14         Sam   75.0     B
5       2    15        Golu   79.0     B
0       1    11      Aditya   85.0     A
8       3    34         Amy   88.0     A
1       1    12       Chris    NaN     A
3       1    15       Harry    NaN   NaN
7       2    23       Clara    NaN     B
9       3    15    Prashant    NaN     B
11      3    23  Radheshyam    NaN   NaN
The 'Marks' column is sorted: False
```

即使我们将具有 NaN 值的行放在数据帧的顶部，`is_monotonic_increasing`属性的计算结果也将为 False。

```py
import pandas as pd
df=pd.read_csv("grade.csv")
df.sort_values(by="Marks",inplace=True,ascending=True,na_position="first")
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_increasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
1       1    12       Chris    NaN     A
3       1    15       Harry    NaN   NaN
7       2    23       Clara    NaN     B
9       3    15    Prashant    NaN     B
11      3    23  Radheshyam    NaN   NaN
6       2    27       Harsh   55.0     C
10      3    27      Aditya   55.0     C
4       2    22         Tom   73.0     B
2       1    14         Sam   75.0     B
5       2    15        Golu   79.0     B
0       1    11      Aditya   85.0     A
8       3    34         Amy   88.0     A
The 'Marks' column is sorted: False
```

### 检查熊猫数据框中的列是否按降序排列

为了检查 pandas 数据帧中的列是否按降序排序，我们将使用`is_monotonic_decreasing`属性。如果按降序对列进行排序，则`is_monotonic_decreasing`属性的计算结果为 True。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
df.sort_values(by="Marks",inplace=True,ascending=False)
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_decreasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
3      3    34         Amy     88     A
2      3    33        Tina     82     A
1      2    23       Clara     78     B
4      3    15    Prashant     78     B
6      3    23  Radheshyam     78     B
0      2    27       Harsh     55     C
5      3    27      Aditya     55     C
7      3    11       Bobby     50     D
The 'Marks' column is sorted: True
```

如果一列未排序或按升序排序，则`is_monotonic_decreasing`属性的计算结果为 False，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
#df.sort_values(by="Marks",inplace=True,ascending=False)
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_decreasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
The 'Marks' column is sorted: False
```

`is_monotonic_decreasing`不能用于具有 NaN 值的列。如果一个列有 NaN 值，那么`is_monotonic_decreasing`属性的计算结果总是 False。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade.csv")
df.sort_values(by="Marks",inplace=True,ascending=False,na_position="last")
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_decreasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
8       3    34         Amy   88.0     A
0       1    11      Aditya   85.0     A
5       2    15        Golu   79.0     B
2       1    14         Sam   75.0     B
4       2    22         Tom   73.0     B
6       2    27       Harsh   55.0     C
10      3    27      Aditya   55.0     C
1       1    12       Chris    NaN     A
3       1    15       Harry    NaN   NaN
7       2    23       Clara    NaN     B
9       3    15    Prashant    NaN     B
11      3    23  Radheshyam    NaN   NaN
The 'Marks' column is sorted: False
```

即使我们将具有 NaN 值的行放在数据帧的顶部，`is_monotonic_decreasing`属性的计算结果也将为 False。

```py
import pandas as pd
df=pd.read_csv("grade.csv")
df.sort_values(by="Marks",inplace=True,ascending=False,na_position="first")
print("The dataframe is:")
print(df)
temp=df["Marks"].is_monotonic_decreasing
print("The 'Marks' column is sorted:",temp)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
1       1    12       Chris    NaN     A
3       1    15       Harry    NaN   NaN
7       2    23       Clara    NaN     B
9       3    15    Prashant    NaN     B
11      3    23  Radheshyam    NaN   NaN
8       3    34         Amy   88.0     A
0       1    11      Aditya   85.0     A
5       2    15        Golu   79.0     B
2       1    14         Sam   75.0     B
4       2    22         Tom   73.0     B
6       2    27       Harsh   55.0     C
10      3    27      Aditya   55.0     C
The 'Marks' column is sorted: False
```

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能还会喜欢这篇关于用 Python](https://codinginfinite.com/regression-in-machine-learning-with-examples/) 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## 使用 Numpy 模块检查数据帧中的列是否排序

python 中的 numpy 模块为我们提供了不同的函数来对数值数据执行操作。一个这样的函数是`diff()`函数。`diff()`函数将 iterable 对象作为其输入参数，并返回一个包含数组元素的一阶差的数组，如下例所示。

```py
import numpy as np
df=pd.read_csv("grade2.csv")
marks=df["Marks"]
print("The Marks column is:")
print(marks)
temp=np.diff(marks)
print("Array returned by diff() is:")
print(temp)
```

输出:

```py
The Marks column is:
0    55
1    78
2    82
3    88
4    78
5    55
6    78
7    50
Name: Marks, dtype: int64
Array returned by diff() is:
[ 23   4   6 -10 -23  23 -28]
```

在这里，您可以观察到一阶差被计算为输入数组中`(n+1)th`和第 n 个元素的差。例如，输出数组的第一个元素是输入`"Marks"`列的第二个元素和第一个元素之差。输出数组中的第二个元素是第三个元素和`"Marks"`列的第二个元素的差。

通过观察输出，我们可以得出结论“如果‘Marks’列按升序排序，输出数组中的所有值都将大于或等于 0。同样，如果‘marks’列按降序排序，输出数组中的所有元素都将小于或等于 0。我们将使用这个结论来检查该列是按升序还是降序排序的。

要检查 pandas 数据帧的一列是否按升序排序，我们将使用以下步骤。

*   首先，我们将计算指定列的一阶差分。为此，我们将把列作为输入参数传递给`diff()` 函数。
*   之后，我们将检查输出数组中的所有元素是否都小于或等于 0。为此，我们将使用比较运算符和`all()`方法。当我们在 numpy 数组上使用比较运算符时，我们会得到一个布尔值数组。在包含布尔值的数组上调用 `all()`方法时，如果所有元素都为真，则返回`True`。
*   如果`all()` 方法返回 True，它将断定所有的元素都按升序排序。

您可以在下面的示例中观察到这一点。

```py
import numpy as np
df=pd.read_csv("grade2.csv")
df.sort_values(by="Marks",inplace=True,ascending=True)
marks=df["Marks"]
print("The dataframe is:")
print(df)
temp=np.diff(marks)
print("Array returned by diff() is:")
print(temp)
boolean_array= temp>=0
print("Boolean array is:")
print(boolean_array)
result=boolean_array.all()
if result:
    print("The marks column is sorted.")
else:
    print("The marks column is not sorted.")
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
7      3    11       Bobby     50     D
0      2    27       Harsh     55     C
5      3    27      Aditya     55     C
1      2    23       Clara     78     B
4      3    15    Prashant     78     B
6      3    23  Radheshyam     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
Array returned by diff() is:
[ 5  0 23  0  0  4  6]
Boolean array is:
[ True  True  True  True  True  True  True]
The marks column is sorted.
```

为了检查一列是否按降序排序，我们将检查`diff()` 函数的输出数组中的所有元素是否都小于或等于 0。为此，我们将使用比较运算符和`all()`方法。当我们在 numpy 数组上使用比较运算符时，我们会得到一个布尔值数组。在包含布尔值的数组上调用`all()` 方法时，如果所有元素都为真，则返回真。

如果`all()`方法返回 True，它将得出结论，所有元素都按降序排序。您可以在下面的示例中观察到这一点。

```py
import numpy as np
df=pd.read_csv("grade2.csv")
df.sort_values(by="Marks",inplace=True,ascending=False)
marks=df["Marks"]
print("The dataframe is:")
print(df)
temp=np.diff(marks)
print("Array returned by diff() is:")
print(temp)
boolean_array= temp<=0
print("Boolean array is:")
print(boolean_array)
result=boolean_array.all()
if result:
    print("The marks column is sorted.")
else:
    print("The marks column is not sorted.")
```

输出:

```py
The dataframe is:
   Class  Roll        Name  Marks Grade
3      3    34         Amy     88     A
2      3    33        Tina     82     A
1      2    23       Clara     78     B
4      3    15    Prashant     78     B
6      3    23  Radheshyam     78     B
0      2    27       Harsh     55     C
5      3    27      Aditya     55     C
7      3    11       Bobby     50     D
Array returned by diff() is:
[ -6  -4   0   0 -23   0  -5]
Boolean array is:
[ True  True  True  True  True  True  True]
The marks column is sorted.
```

## 检查数据帧中的索引列是否已排序

要检查数据帧的索引是否按升序排序，我们可以使用如下所示的`index`属性和`is_monotonic`属性。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
df.sort_index(inplace=True,ascending=True)
print("The dataframe is:")
print(df)
temp=df.index.is_monotonic
print("The Index is sorted:",temp)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
50         3    11       Bobby     D
55         2    27       Harsh     C
55         3    27      Aditya     C
78         2    23       Clara     B
78         3    15    Prashant     B
78         3    23  Radheshyam     B
82         3    33        Tina     A
88         3    34         Amy     A
The Index is sorted: True
```

要检查数据帧的[索引是否按升序排序，我们可以使用 index 属性和`is_monotonic_increasing`属性，如下所示。](https://www.pythonforbeginners.com/basics/pandas-dataframe-index-in-python)

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
df.sort_index(inplace=True,ascending=True)
print("The dataframe is:")
print(df)
temp=df.index.is_monotonic_increasing
print("The Index is sorted:",temp)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
50         3    11       Bobby     D
55         2    27       Harsh     C
55         3    27      Aditya     C
78         2    23       Clara     B
78         3    15    Prashant     B
78         3    23  Radheshyam     B
82         3    33        Tina     A
88         3    34         Amy     A
The Index is sorted: True
```

要检查数据帧的索引是否按降序排序，我们可以使用如下所示的`index`属性和`is_monotonic_decreasing`属性。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
df.sort_index(inplace=True,ascending=False)
print("The dataframe is:")
print(df)
temp=df.index.is_monotonic_decreasing
print("The Index is sorted:",temp)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
88         3    34         Amy     A
82         3    33        Tina     A
78         2    23       Clara     B
78         3    15    Prashant     B
78         3    23  Radheshyam     B
55         2    27       Harsh     C
55         3    27      Aditya     C
50         3    11       Bobby     D
The Index is sorted: True
```

您需要记住，如果索引列包含 NaN 值，那么`is_monotonic`属性、`is_monotonic_increasing`属性和`is_monotonic_decreasing`总是返回 False。因此，如果索引列包含 NaN 值，则不能使用这些属性来检查索引是否已排序。

## 结论

在本文中，我们讨论了检查 pandas 数据帧中的列是否排序的不同方法。为此，我们使用了 pandas 库以及 [numpy 模块](https://www.pythonforbeginners.com/basics/create-numpy-array-in-python)。我们还检查了熊猫数据帧的索引是否排序。

要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[字典理解的文章。你可能也会喜欢这篇关于 python](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 中的[列表理解的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！