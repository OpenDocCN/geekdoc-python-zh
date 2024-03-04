# 从 Pandas 数据框架中删除列

> 原文：<https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe>

在 python 中处理数据帧时，我们经常需要在数据预处理时从数据帧中删除一个或多个列。在本文中，我们将讨论用 python 从 pandas 数据帧中删除列的不同方法。

## drop()方法

drop()方法可用于从 [pandas 数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)中删除列或行。它具有以下语法。

```py
DataFrame.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
```

这里，

*   当我们必须从数据帧中删除一行时，使用`index`参数。`index`参数将必须删除的一个索引或索引列表作为其输入参数。
*   当我们需要从数据帧中删除一列时，使用`columns`参数。columns 参数将需要删除的列名或列名列表作为其输入参数。
*   `labels`参数表示我们需要从数据帧中移除的索引或列标签。为了从数据帧中删除行，我们使用了`index`标签。为了从数据帧中删除一列，我们使用了`columns`标签。要删除两个或更多的列或行，我们还可以将索引的列名列表分别传递给列和索引标签。
*   当我们不使用 index 和 columns 参数时，我们将需要删除的行的列名或索引传递给`labels`参数作为它的输入参数。在这种情况下，我们使用`axis`参数来决定是删除一行还是一列。如果我们想从数据帧中删除一列，我们将`axis`参数设置为 1。当我们想要从数据帧中删除一行时，我们将`axis`参数设置为 0，这是它的默认值。
*   当我们有多级索引时，`level`参数用于从数据帧中删除行或列。`level`参数接受我们希望从数据帧中删除的列或行的索引级别或索引名称。要删除两个或更多级别，可以将索引级别或索引名称的列表传递给`level`参数。
*   `inplace`参数用于决定我们是在删除操作后获得一个新的数据帧，还是想要修改原始数据帧。当`inplace`被设置为`False`时，这是它的默认值，原始数据帧不变，`drop()`方法在执行后返回修改后的数据帧。要修改原始数据帧，可以将`inplace`设置为`True`。
*   `errors`参数用于决定我们是否希望在执行`drop()` 方法时引发异常和错误。默认情况下，误差参数设置为 `“raise”`。因此，如果在执行过程中出现任何问题，`drop()`方法会引发一个异常。如果不希望出现错误，可以将错误参数设置为`“ignore”`。在这之后，`drop()` 方法将[抑制所有的异常](https://www.pythonforbeginners.com/basics/suppress-exceptions-in-python)。

执行后，如果`inplace`参数设置为`False`，则`drop()` 方法返回修改后的数据帧。否则返回`None`。

## 在 Python 中，熊猫按名称删除列

要按名称从 pandas dataframe 中删除一列，可以将列名传递给参数`labels`，并在`drop()`方法中将参数`axis`设置为 1。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
output_df=grades.drop(labels="Marks",axis=1)
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

在上面的例子中，我们首先使用 `read_csv`()方法从 CSV 文件中读取熊猫数据帧。然后，我们调用输入数据帧上的`drop()`方法，将`Marks" as an` 输入参数设置为`labels`参数，并设置`axis=1`。您可以观察到由`drop()`方法返回的 dataframe 没有`"Marks"`列。

不使用`labels`和`axis`参数，您可以将列名传递给`drop()`方法中的`columns`参数，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
output_df=grades.drop(columns="Marks")
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

在上面的例子中，我们没有使用两个参数，即`labels`和`axis`，而是使用了`columns`参数来从输入数据帧中删除 `"Marks"` 列。

## 从 Pandas 数据框架中按索引删除列

除了使用列名，还可以通过索引从数据帧中删除列。

要通过索引从 pandas 数据帧中删除列，我们首先需要使用数据帧的 columns 属性获得 columns 对象。然后，我们可以对列的索引使用索引操作符，按索引删除列，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
output_df=grades.drop(labels=grades.columns[3],axis=1)
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

在上面的例子中，我们首先使用`grades.columns`属性获得了`grades`数据帧的列名。然后，我们可以获得`grades.columns`属性中索引 3 处的元素。最后，我们已经将值传递给了`labels`参数。您可以在输出中观察到由`drop()`方法返回的数据帧没有输入数据帧的第 4 列。因此，我们使用索引成功地删除了数据帧中的一列。

我们也可以使用`columns`参数代替`labels`参数，如下例所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
output_df=grades.drop(columns=grades.columns[3])
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

### 在 Python 中从数据帧中删除第一列

要删除 dataframe 的第一列，可以将索引 0 处的列名传递给参数`labels`，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
output_df=grades.drop(labels=grades.columns[0],axis=1)
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Roll        Name  Marks Grade
0     11      Aditya   85.0     A
1     12       Chris    NaN     A
2     14         Sam   75.0     B
3     15       Harry    NaN   NaN
4     22         Tom   73.0     B
5     15        Golu   79.0     B
6     27       Harsh   55.0     C
7     23       Clara    NaN     B
8     34         Amy   88.0     A
9     15    Prashant    NaN     B
10    27      Aditya   55.0     C
11    23  Radheshyam    NaN   NaN 
```

在本例中，我们必须在第一个位置删除该列。因此，我们将输入数据帧的`grades.columns`属性的索引 0 处的元素传递给了`labels`参数。由于我们必须删除数据帧中的一列，我们还需要设置`axis=1`。

不使用`labels`和`axis`参数，您可以使用如下所示的`columns`参数。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
output_df=grades.drop(columns=grades.columns[0])
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Roll        Name  Marks Grade
0     11      Aditya   85.0     A
1     12       Chris    NaN     A
2     14         Sam   75.0     B
3     15       Harry    NaN   NaN
4     22         Tom   73.0     B
5     15        Golu   79.0     B
6     27       Harsh   55.0     C
7     23       Clara    NaN     B
8     34         Amy   88.0     A
9     15    Prashant    NaN     B
10    27      Aditya   55.0     C
11    23  Radheshyam    NaN   NaN
```

在这个例子中，我们已经将位于`grades.columns` 属性索引 0 的元素传递给了`drop()`方法中的`columns`参数。`grades.columns`属性本质上包含一个列名列表。因此，使用列的索引删除列的过程类似于使用列的名称。它只是有一些额外的计算。因此，如果我们知道数据帧中列的名称，那么直接按列名删除列总是更好。

### 从熊猫数据框架中删除最后一列

为了删除数据帧的最后一列，我们将首先找到最后一列的索引。为此，我们将找到数据帧的列属性的长度。在此之后，我们将通过从长度中减去 1 来找到最后一列的索引。然后，我们将使用最后一列的索引和`columns`属性获得最后一列的名称。最后，我们将把获得的列名传递给`labels`参数。执行后，`drop()`方法将删除数据帧的最后一列。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
lenDf=len(grades.columns)
output_df=grades.drop(labels=grades.columns[lenDf-1],axis=1)
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name  Marks
0       1    11      Aditya   85.0
1       1    12       Chris    NaN
2       1    14         Sam   75.0
3       1    15       Harry    NaN
4       2    22         Tom   73.0
5       2    15        Golu   79.0
6       2    27       Harsh   55.0
7       2    23       Clara    NaN
8       3    34         Amy   88.0
9       3    15    Prashant    NaN
10      3    27      Aditya   55.0
11      3    23  Radheshyam    NaN
```

不使用`labels`参数，您可以使用 columns 参数从 pandas 数据帧中删除列，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
lenDf=len(grades.columns)
output_df=grades.drop(columns=grades.columns[lenDf-1])
print("The output dataframe is:")
print(output_df)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name  Marks
0       1    11      Aditya   85.0
1       1    12       Chris    NaN
2       1    14         Sam   75.0
3       1    15       Harry    NaN
4       2    22         Tom   73.0
5       2    15        Golu   79.0
6       2    27       Harsh   55.0
7       2    23       Clara    NaN
8       3    34         Amy   88.0
9       3    15    Prashant    NaN
10      3    27      Aditya   55.0
11      3    23  Radheshyam    NaN
```

## 熊猫在 Python 中将列放到适当的位置

默认情况下,`drop()`方法不会修改原始数据帧。删除指定的列后，它返回修改后的数据帧。

为了从原始数据帧中删除列，我们将把参数`inplace`设置为`True`。此后，`drop()` 方法修改原始数据帧，而不是返回一个新的数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.drop(columns="Marks",inplace=True)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

在本例中，我们已经将`inplace`参数设置为`True`。结果，`drop()`方法修改了原始数据帧。

## 如果在 Python 中存在，熊猫会删除列

如果数据帧中不存在列名，`drop()` 方法会引发一个 [KeyError 异常](https://www.pythonforbeginners.com/basics/python-keyerror)。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.drop(columns="Height",inplace=True)
print("The output dataframe is:")
print(grades)
```

输出

```py
KeyError: "['Height'] not found in axis"
```

在这个例子中，`drop()`方法的 `"Height" parameter that is given as an` 输入参数没有作为 dataframe 中的列出现。因此，`drop()`方法会引发 KeyError 异常。

为了删除存在的列而不遇到异常，我们将在`drop()`方法中使用`errors`参数。您可以将误差参数设置为`“ignore”`。在这之后，`drop()`方法将抑制所有的异常。如果输入数据帧包含指定的列，它将正常执行。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.drop(columns="Marks",inplace=True,errors="ignore")
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

在上面的例子中，`"Marks"` 列出现在 dataframe 中。因此，它在执行`drop()`方法时被删除。

如果 columns 参数中指定的列在 dataframe 中不存在，`drop()`方法什么也不做。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.drop(columns="Height",inplace=True,errors="ignore")
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
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
```

在上面的例子中，作为`drop()` 方法的输入参数给出的`"Height"`参数没有作为 dataframe 中的列出现。因此，`drop()`方法对原始数据帧没有影响。

## 从熊猫数据框架中删除多列

要从 pandas 数据帧中删除多个列，可以将列名列表传递给 labels 参数，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.drop(labels=["Marks", "Grade"],axis=1,inplace=True)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name
0       1    11      Aditya
1       1    12       Chris
2       1    14         Sam
3       1    15       Harry
4       2    22         Tom
5       2    15        Golu
6       2    27       Harsh
7       2    23       Clara
8       3    34         Amy
9       3    15    Prashant
10      3    27      Aditya
11      3    23  Radheshyam
```

在上面的例子中，我们已经将列表 `["Marks", "Grade"]` 传递给了参数`labels`。因此，`drop()`方法从输入数据帧中删除了`Marks`和`Grade`列。

代替`labels`参数，您可以使用`columns`参数从 pandas 数据帧中删除多个列，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.drop(columns=["Marks", "Grade"],inplace=True)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name
0       1    11      Aditya
1       1    12       Chris
2       1    14         Sam
3       1    15       Harry
4       2    22         Tom
5       2    15        Golu
6       2    27       Harsh
7       2    23       Clara
8       3    34         Amy
9       3    15    Prashant
10      3    27      Aditya
11      3    23  Radheshyam
```

您还可以使用列的索引从 pandas 数据帧中删除多个列，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.drop(columns=grades.columns[3:5],inplace=True)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name
0       1    11      Aditya
1       1    12       Chris
2       1    14         Sam
3       1    15       Harry
4       2    22         Tom
5       2    15        Golu
6       2    27       Harsh
7       2    23       Clara
8       3    34         Amy
9       3    15    Prashant
10      3    27      Aditya
11      3    23  Radheshyam
```

在这个例子中，我们使用了[列表切片](https://www.pythonforbeginners.com/dictionary/python-slicing)来获得数据帧的第三和第四个位置的列名。与使用列名相比，使用列索引来删除数据帧的成本更高。但是，当我们想要删除前 n 列或后 n 列，或者在没有列名的情况下从特定索引中删除列时，这是很有用的。

### 从熊猫数据框架中删除前 N 列

为了删除数据帧的前 n 列，我们将使用数据帧的`columns`属性获得 columns 对象。然后，我们可以使用带有列索引的索引操作符从数据帧中删除前 n 列，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
n=2
grades.drop(columns=grades.columns[:n],inplace=True)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
          Name  Marks Grade
0       Aditya   85.0     A
1        Chris    NaN     A
2          Sam   75.0     B
3        Harry    NaN   NaN
4          Tom   73.0     B
5         Golu   79.0     B
6        Harsh   55.0     C
7        Clara    NaN     B
8          Amy   88.0     A
9     Prashant    NaN     B
10      Aditya   55.0     C
11  Radheshyam    NaN   NaN
```

在上面的例子中，我们获取了`columns`属性的一部分，以获得索引为 0 和 1 的列的列名。然后，我们将切片传递给`drop()`方法中的 columns 参数来删除列。

### 从熊猫数据框架中删除最后 N 列

为了删除数据帧的最后 n 列，我们将使用数据帧的`columns`属性获得 columns 对象。然后，我们可以使用带有列索引的索引操作符从数据帧中删除最后 n 列，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
lenDf=len(grades.columns)
n=2
grades.drop(columns=grades.columns[lenDf-n:lenDf],inplace=True)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name
0       1    11      Aditya
1       1    12       Chris
2       1    14         Sam
3       1    15       Harry
4       2    22         Tom
5       2    15        Golu
6       2    27       Harsh
7       2    23       Clara
8       3    34         Amy
9       3    15    Prashant
10      3    27      Aditya
11      3    23  Radheshyam
```

在上面的例子中，我们已经计算了 dataframe 中的列总数，并将其存储在`lenDf`变量中。然后，我们使用列表切片从`grades.columns` 列表中获得最后 n 列。获得最后 n 列的名称后，我们将它传递给 columns 参数，以便从 pandas 数据帧中删除最后 n 列。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能还会喜欢这篇关于用 Python](https://codinginfinite.com/regression-in-machine-learning-with-examples/) 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## dropna()方法

`dropna()` 方法可用于删除具有 nan 值的列。它具有以下语法。

```py
DataFrame.dropna(*, axis=0, how=_NoDefault.no_default, thresh=_NoDefault.no_default, subset=None, inplace=False)
```

这里，

*   `axis`参数用于决定我们是否想要删除具有 nan 值的行或列。默认情况下，`axis`设置为 0。因此，当在数据帧上执行`dropna()` 方法时，带有 nan 值的行将被删除。要删除具有 nan 值的列，可以将`axis`参数设置为 1。
*   `how`参数用于确定需要删除的列是否应该包含所有的值`NaN`，或者是否可以因为包含至少一个`NaN`值而将其删除。默认情况下，`how`参数设置为`“any”`。因此，即使只存在一个 nan，也将从数据帧中删除该列。
*   当我们希望删除至少有特定数量的非 NaN 值的列时，使用`thresh`参数。例如，如果您想要删除一个少于 n 个非 NaN 值的列，您可以将数字 n 传递给`thresh`参数。
*   当我们想要检查每一列的特定索引中的`NaN`值时，使用`subset`参数。默认情况下，子集参数设置为`None`。因此， `dropna()`方法在所有索引中搜索`NaN`值。如果希望它只搜索特定行中的 nan 值，可以将行索引传递给`subset`参数。要检查两行或更多行中的 nan 值，可以将索引列表传递给`subset`参数。
*   `inplace`参数用于决定我们是在删除操作后获得一个新的数据帧，还是想要修改原始数据帧。当`inplace`被设置为`False`时，这是它的默认值，原始数据帧不变，`dropna()`方法在执行后返回修改后的数据帧。要修改原始数据帧，可以将`inplace`设置为`True`。

执行后，如果`inplace`设置为`False`，则`dropna()`方法返回修改后的数据帧。否则返回`None`。

## 在 Pandas 数据框架中删除带有 NaN 值的列

要删除具有 nan 值的列，可以在输入数据帧上调用`dropna()` 方法。此外，您需要将`axis`参数设置为 1。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.dropna(axis=1,inplace=True)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name
0       1    11      Aditya
1       1    12       Chris
2       1    14         Sam
3       1    15       Harry
4       2    22         Tom
5       2    15        Golu
6       2    27       Harsh
7       2    23       Clara
8       3    34         Amy
9       3    15    Prashant
10      3    27      Aditya
11      3    23  Radheshyam
```

在上面的例子中，`Marks`和`Grades`列在输入数据帧中有`NaN`值。因此，在执行`dropna()`方法后，它们已经从数据帧中删除。

## 删除数据帧中至少有 N 个 NaN 值的列

要删除至少有 n 个 nan 值的列，可以在`dropna()`方法中使用`thresh`参数和`axis`参数。

`thresh`参数将非 NaN 元素的最小数量作为其输入参数。如果与 thresh 参数中指定的值相比，dataframe 中任何列的非 NaN 值的数量较少，则在执行`dropna()` 方法后，该列将从 dataframe 中删除。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
lenDf=len(grades.index)
n=5
count=lenDf-n+1
grades.dropna(axis=1,inplace=True,thresh=count)
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

在这个例子中，如果一列至少有 n 个 NaN 值，我们需要从数据帧中删除它。因此，如果一个列必须包含在输出数据帧中，它应该至少有(数据帧中的行数-n +1)个非 NaN 值。如果任何列的非 NaN 值小于(dataframe 中的行数-n +1 ),则该列将从数据帧中删除。

## 使用 pop()方法从数据帧中删除列

`pop()` 方法可以用来一次从数据帧中删除一列。它具有以下语法。

```py
DataFrame.pop(item)
```

在 dataframe 上调用`pop()`方法时，它将一个列名作为输入，并从原始 dataframe 中删除该列。它还将删除的列作为输出返回。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.pop("Marks")
print("The output dataframe is:")
print(grades)
```

输出:

```py
The input dataframe is:
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
The output dataframe is:
    Class  Roll        Name Grade
0       1    11      Aditya     A
1       1    12       Chris     A
2       1    14         Sam     B
3       1    15       Harry   NaN
4       2    22         Tom     B
5       2    15        Golu     B
6       2    27       Harsh     C
7       2    23       Clara     B
8       3    34         Amy     A
9       3    15    Prashant     B
10      3    27      Aditya     C
11      3    23  Radheshyam   NaN
```

如果 dataframe 中不存在该列，pop()方法将引发一个 KeyError 异常。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is:")
print(grades)
grades.pop("Height")
print("The output dataframe is:")
print(grades)
```

输出:

```py
KeyError: 'Height'
```

这里,“高度”参数不在数据帧中。因此，pop()方法引发了 [KeyError 异常](https://www.pythonforbeginners.com/basics/python-keyerror)。

## 结论

在本文中，我们讨论了用 Python 从 pandas 数据帧中删除列的不同方法。为此，我们使用了`drop()`方法、 `dropna()`方法和`pop()` 方法。

要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[字典理解的文章。你可能也会喜欢这篇关于 python](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 中的[列表理解的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)