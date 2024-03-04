# 在 Pandas 数据帧中删除带有 Nan 值的行

> 原文：<https://www.pythonforbeginners.com/basics/drop-rows-with-nan-values-in-a-pandas-dataframe>

清理数据时，处理 nan 值是一项繁琐的任务。在本文中，我们将讨论使用`dropna()`方法从 pandas 数据帧中删除具有 nan 值的行的不同方法。

## dropna()方法

`dropna()`方法可用于删除 pandas 数据帧中具有 nan 值的行。它具有以下语法。

```py
DataFrame.dropna(*, axis=0, how=_NoDefault.no_default, thresh=_NoDefault.no_default, subset=None, inplace=False)
```

这里，

*   `axis`参数用于决定我们是否想要删除具有 nan 值的行或列。默认情况下，`axis`参数设置为 0。因此，当在数据帧上执行`dropna()`方法时，带有 nan 值的行将被删除。
*   `“how”`参数用于确定需要删除的行是否应该将所有值都作为 NaN，或者是否可以因为至少有一个 NaN 值而将其删除。默认情况下，`“how”`参数设置为`“any”`。因此，即使存在单个 nan 值，也将从数据帧中删除该行。
*   当我们希望删除至少有特定数量的非 NaN 值的行时，使用`thresh`参数。例如，如果您想要删除一个少于 n 个非空值的行，您可以将数字 n 传递给`thresh`参数。
*   当我们希望只检查每行中特定列的 NaN 值时，使用`subset`参数。默认情况下，`subset`参数设置为无。因此，`dropna()`方法在所有列中搜索 NaN 值。如果您希望它只在每行的特定列中搜索 nan 值，您可以将列名传递给`subset`参数。要检查两列或更多列中的 nan 值，可以将列名列表传递给`subset`参数。
*   `inplace`参数用于决定我们是在删除操作后获得一个新的数据帧，还是想要修改原始数据帧。当`inplace`被设置为 False(这是它的默认值)时，原始数据帧不会改变，dropna()方法在执行后返回修改后的数据帧。要修改原始数据帧，您可以将`inplace`设置为真。

## 删除数据帧中任何列中具有 NaN 值的行

要从一个 [pandas 数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)中删除任何一列中有 nan 值的行，可以直接调用输入数据帧上的`dropna()`方法。执行后，它返回一个修改后的数据帧，其中删除了 nan 值。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df=df.dropna()
print("After dropping NaN values:")
print(df)
```

输出:

```py
The dataframe is:
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
After dropping NaN values:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
3     3.0  34.0         Amy   88.0     A
5     3.0  27.0      Aditya   55.0     C
7     3.0  23.0  Radheshyam   78.0     B
10    3.0  15.0      Lokesh   88.0     A
```

在上面的示例中，输入数据帧包含许多具有 NaN 值的行。一旦我们在输入数据帧上调用了`dropna()`方法，它将返回一个没有空值的数据帧。

## 删除数据帧中所有列中具有 NaN 值的行

默认情况下，如果 dataframe 中至少有一列有 NaN 值，则 `dropna()`方法会删除其中的行。如果您只想删除所有列中都有 NaN 值的数据帧，可以将`dropna()`方法中的`“how”`参数设置为`“all”`。此后，只有当任何一行中的所有列都包含 NaN 值时，才从数据帧中删除这些行。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df=df.dropna(how="all")
print("After dropping NaN values:")
print(df)
```

输出:

```py
The dataframe is:
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
After dropping NaN values:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
2     3.0  33.0         NaN    NaN   NaN
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
10    3.0  15.0      Lokesh   88.0     A
```

在这个例子中，我们已经在`dropna()`方法中将 how 参数设置为`"all"`。因此，只有那些所有值都为空的行才会从输入数据帧中删除。因此，只有在所有列中具有 NaN 值的两行被从输入数据帧中删除，而不是前面例子中观察到的五行。

## 删除至少 N 列中具有非空值的行

您可能还想控制每行中 nan 值的数量，而不是一个或全部。为此，您可以使用`dropna()` 方法中的`thresh`参数指定输出数据帧中每行非空值的最小数量。在这之后，由`dropna()`方法返回的输出数据帧将在每一行中包含至少 N 个空值。这里，N 是作为输入参数传递给 thresh 参数的数字。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df=df.dropna(thresh=4)
print("After dropping NaN values:")
print(df)
```

输出:

```py
The dataframe is:
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
After dropping NaN values:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
10    3.0  15.0      Lokesh   88.0     A
```

在这个例子中，我们在`dropna()`方法中指定了参数`thresh=4`。因此，只有那些具有少于 4 个非空值的行才会从输入数据帧中删除。即使一行有一个空值并且有 4 个以上的非空值，它也不会从数据帧中删除。

## 删除 Pandas 数据帧中至少有 N 个空值的行

您可能希望从输入数据帧中删除具有 N 个以上空值的所有行，而不是在每行中至少保留 N 个非空值。为此，我们将首先使用 columns 属性和`len()` 函数找到输入数据帧中的列数。接下来，我们将从数据帧的总列数中减去 N。结果数将是我们希望在输出数据帧中出现的最少数量的非空值。因此，我们将把这个数字传递给`dropna()`方法中的`thresh`参数。

在执行了`dropna()` 方法之后，我们将在删除每一行中至少有 n 个空值的所有行之后得到输出数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
N=3
number_of_columns=len(df.columns)
df=df.dropna(thresh=number_of_columns-N+1)
print("After dropping NaN values:")
print(df)
```

输出:

```py
The dataframe is:
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
After dropping NaN values:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
10    3.0  15.0      Lokesh   88.0     A
```

这个例子只是前一个例子的变体。如果要删除超过 N 个空值的行，需要保留列数为 N+1 或更多非空值的行。这就是我们在这个例子中所做的。

## 删除 Pandas 中特定列中具有 NaN 值的行

默认情况下， `dropna()`方法在每行的所有列中搜索 NaN 值。如果您希望仅当数据帧的特定列中有空值时才删除数据帧中的行，您可以在`dropna()`方法中使用`subset`参数。

`dropna()` 方法中的`subset`参数将一列列名作为其输入参数。此后，`dropna()`方法只删除指定列中具有空值的行。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df=df.dropna(subset=["Class","Roll","Marks"])
print("After dropping NaN values:")
print(df)
```

输出:

```py
The dataframe is:
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
After dropping NaN values:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
3     3.0  34.0         Amy   88.0     A
4     3.0  15.0         NaN   78.0     B
5     3.0  27.0      Aditya   55.0     C
7     3.0  23.0  Radheshyam   78.0     B
8     3.0  11.0       Bobby   50.0   NaN
10    3.0  15.0      Lokesh   88.0     A
```

在这个例子中，我们将列表 `["Class", "Roll", "Marks"]`传递给了`dropna()`方法中的`subset`参数。因此，`dropna()`方法仅在数据帧的这些列中搜索 NaN 值。在执行`dropna()`方法后，这些列中任何具有 NaN 值的行都将从数据帧中删除。如果一行在这些列中有非空值，并且在其他列中有 NaN 值，则不会将其从数据帧中删除。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇 [MLFlow 教程，里面有代码示例](https://codinginfinite.com/mlflow-tutorial-with-code-example/)。你可能也会喜欢这篇关于 2023 年 T2 15 款免费数据可视化工具的文章。

## 从 Pandas 数据帧中删除具有 NaN 值的行

在前面几节的所有例子中，`dropna()`方法不修改输入数据帧。每次，它都返回一个新的数据帧。要通过删除 nan 值来修改输入数据帧，可以使用`dropna()`方法中的`inplace`参数。当`inplace`参数设置为真时，`dropna()`方法修改原始数据帧，而不是创建一个新的数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df.dropna(inplace=True)
print("After dropping NaN values:")
print(df)
```

输出:

```py
The dataframe is:
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
After dropping NaN values:
    Class  Roll        Name  Marks Grade
0     2.0  27.0       Harsh   55.0     C
1     2.0  23.0       Clara   78.0     B
3     3.0  34.0         Amy   88.0     A
5     3.0  27.0      Aditya   55.0     C
7     3.0  23.0  Radheshyam   78.0     B
10    3.0  15.0      Lokesh   88.0     A
```

在这个例子中，我们已经在`dropna()`方法中将`inplace`参数设置为 True。因此，`dropna()` 方法修改原始数据帧，而不是创建一个新的数据帧。

## 结论

在本文中，我们讨论了使用`dropna()`方法从 pandas 数据帧中删除具有 NaN 值的行的不同方法。

要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！