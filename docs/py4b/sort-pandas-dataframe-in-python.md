# 用 Python 对熊猫数据帧进行排序

> 原文：<https://www.pythonforbeginners.com/basics/sort-pandas-dataframe-in-python>

Pandas 数据帧用于在 Python 中处理表格数据。很多时候，我们需要根据列对数据帧进行排序。在本文中，我们将讨论在 Python 中对熊猫数据帧进行排序的不同方法。

## sort_values()方法

sort_values()函数用于对一个[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)进行水平或垂直排序。它具有以下语法。

```py
DataFrame.sort_values(by, *, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)
```

这里，

*   `by`参数将一个字符串或一系列字符串作为其输入参数。`by`参数的输入取决于我们是否想要对数据帧的行或列进行排序。为了根据列对数据帧的行进行排序，我们可以将一个列名或列名列表传递给`by`参数。为了根据行对数据帧的列进行排序，我们可以将行索引或行索引列表传递给参数`by`。
*   `axis`参数用于决定我们是否要对数据帧的行或列进行排序。为了根据一列或一组列对数据帧中的行进行排序，我们可以将值 0 传递给参数`axis`，这是它的默认值。为了根据一行或多行对数据帧的列进行排序，我们可以将值 1 传递给`axis`参数。
*   `ascending`参数用于决定数据帧是按升序还是降序排序。默认情况下，`True`表示按升序排序。您可以将它设置为`False`，以降序对数据帧进行排序。如果排序是由多列完成的，您可以将一列`True`和`False`值传递给`ascending`参数，以决定数据帧是按升序还是降序排序。
*   `inplace`参数用于决定我们是修改原始数据帧还是在排序后创建新的数据帧。默认情况下，`inplace`设置为`False`。因此，它不会修改原始数据帧，而`sort_values()`方法会返回新排序的数据帧。如果您想在排序时修改原始数据帧，可以将`inplace`设置为`True`。
*   `kind`参数用于决定排序算法。默认情况下，`sort_values()`方法使用快速排序算法。经过数据分析，如果你认为输入的数据有一个确定的模式，某种排序算法可以减少时间，你可以使用`‘mergesort’`、`‘heapsort’`或`‘stable’` 排序算法。
*   `na_position`参数用于决定具有`NaN`值的行的位置。默认情况下，它的值为`'last'`,表示具有`NaN`值的行最终存储在排序后的数据帧中。如果您想让带有`NaN`值的行位于排序数据帧的顶部，您可以将它设置为`“first”` 。
*   `ignore_index`参数用于决定输入数据帧中行的索引是否保留在排序数据帧中。默认情况下，`True`表示索引被保留。如果您想忽略初始数据帧的索引，您可以将`ignore_index`设置为`True`。
*   `key`参数用于在排序前对数据帧的列执行操作。它将一个矢量化函数作为其输入参数。提供给`key`参数的函数必须将熊猫系列作为其输入参数，并返回熊猫系列。在排序之前，该函数独立应用于输入数据帧中的每一列。

执行后，如果 inplace 参数设置为`False`，则`sort_values()` 方法返回排序后的数据帧。如果`inplace`被设置为`True`，则`sort_values()`方法返回`None`。

## Python 中按列对数据帧的行进行排序

为了按列对数据帧进行排序，我们将在数据帧上调用`sort_values()`方法。我们将把数据帧排序所依据的列名作为输入参数传递给`“by”`参数。执行后， `sort_values()`方法将返回排序后的数据帧。下面是我们在本文中用来创建数据帧的 CSV 文件。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
sorted_df=grades.sort_values(by="Marks")
print("The sorted dataframe is")
print(sorted_df)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya     85     A
1       1    12       Chris     95     A
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
4       1    15       Harry     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
7       2    15        Golu     79     B
8       2    27       Harsh     55     C
9       2    23       Clara     78     B
10      3    33        Tina     82     A
11      3    34         Amy     88     A
12      3    15    Prashant     78     B
13      3    27      Aditya     55     C
14      3    23  Radheshyam     78     B
15      3    11       Bobby     50     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
15      3    11       Bobby     50     D
4       1    15       Harry     55     C
8       2    27       Harsh     55     C
13      3    27      Aditya     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
9       2    23       Clara     78     B
12      3    15    Prashant     78     B
14      3    23  Radheshyam     78     B
7       2    15        Golu     79     B
10      3    33        Tina     82     A
0       1    11      Aditya     85     A
11      3    34         Amy     88     A
1       1    12       Chris     95     A
```

在上面的例子中，我们首先[使用`read_csv()`函数将 CSV 文件](https://www.pythonforbeginners.com/basics/read-specific-columns-from-csv-file)读入数据帧。`read_csv()`函数获取 CSV 文件的文件名并返回一个数据帧。获得数据帧后，我们使用`sort_values()`方法通过`"Marks"`对其进行排序。

这里，`sort_values()` 返回一个新的排序数据帧。如果想对原始数据帧进行排序，可以使用如下所示的`inplace=True`参数。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by="Marks",inplace=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya     85     A
1       1    12       Chris     95     A
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
4       1    15       Harry     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
7       2    15        Golu     79     B
8       2    27       Harsh     55     C
9       2    23       Clara     78     B
10      3    33        Tina     82     A
11      3    34         Amy     88     A
12      3    15    Prashant     78     B
13      3    27      Aditya     55     C
14      3    23  Radheshyam     78     B
15      3    11       Bobby     50     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
15      3    11       Bobby     50     D
4       1    15       Harry     55     C
8       2    27       Harsh     55     C
13      3    27      Aditya     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
9       2    23       Clara     78     B
12      3    15    Prashant     78     B
14      3    23  Radheshyam     78     B
7       2    15        Golu     79     B
10      3    33        Tina     82     A
0       1    11      Aditya     85     A
11      3    34         Amy     88     A
1       1    12       Chris     95     A
```

您可以观察到在将`inplace`设置为`True`后，原始数据帧已经被排序，

在上面的例子中，索引也随着行被混洗。这有时是不希望的。要通过刷新索引来更改行的索引，可以将`ignore_index`参数设置为`True`。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by="Marks",inplace=True,ignore_index=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya     85     A
1       1    12       Chris     95     A
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
4       1    15       Harry     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
7       2    15        Golu     79     B
8       2    27       Harsh     55     C
9       2    23       Clara     78     B
10      3    33        Tina     82     A
11      3    34         Amy     88     A
12      3    15    Prashant     78     B
13      3    27      Aditya     55     C
14      3    23  Radheshyam     78     B
15      3    11       Bobby     50     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
0       3    11       Bobby     50     D
1       1    15       Harry     55     C
2       2    27       Harsh     55     C
3       3    27      Aditya     55     C
4       2     1        Joel     68     B
5       2    22         Tom     73     B
6       1    14         Sam     75     B
7       1    16      Aditya     78     B
8       2    23       Clara     78     B
9       3    15    Prashant     78     B
10      3    23  Radheshyam     78     B
11      2    15        Golu     79     B
12      3    33        Tina     82     A
13      1    11      Aditya     85     A
14      3    34         Amy     88     A
15      1    12       Chris     95     A
```

在上面的示例中，您可以观察到每个位置的行索引与原始数据帧相同，并且没有与输入行混排。这是由于我们指定了`ignore_index`到`True`的原因。

## 按多列对数据帧的行进行排序

除了按一列对数据帧进行排序，我们还可以按多列对数据帧的行进行排序。

要按多列对 pandas 数据帧的行进行排序，可以将列名列表作为输入参数传递给`“by”`参数。当我们传递一个列名列表时，根据第一个元素对行进行排序。之后，根据列表的第二个元素对它们进行排序，以此类推。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by=["Class","Marks"],inplace=True,ignore_index=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya     85     A
1       1    12       Chris     95     A
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
4       1    15       Harry     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
7       2    15        Golu     79     B
8       2    27       Harsh     55     C
9       2    23       Clara     78     B
10      3    33        Tina     82     A
11      3    34         Amy     88     A
12      3    15    Prashant     78     B
13      3    27      Aditya     55     C
14      3    23  Radheshyam     78     B
15      3    11       Bobby     50     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
0       1    15       Harry     55     C
1       1    14         Sam     75     B
2       1    16      Aditya     78     B
3       1    11      Aditya     85     A
4       1    12       Chris     95     A
5       2    27       Harsh     55     C
6       2     1        Joel     68     B
7       2    22         Tom     73     B
8       2    23       Clara     78     B
9       2    15        Golu     79     B
10      3    11       Bobby     50     D
11      3    27      Aditya     55     C
12      3    15    Prashant     78     B
13      3    23  Radheshyam     78     B
14      3    33        Tina     82     A
15      3    34         Amy     88     A
```

在上面的例子中，我们按照两列对数据帧进行了排序，即`Class`和`Marks`。为此，我们将列表`["Class", "Marks"]` 传递给了`sort_values()`方法中的`by`参数。

这里，dataframe 按照`by`参数中列名的顺序排序。首先，dataframe 按`"Class"` 列排序。当两个或多个行在`"Class"`列中具有相同的值时，这些行将按`"Marks"` 列排序。

## 在熊猫数据帧中按降序排列值

默认情况下，`sort_values()` 方法按升序对数据帧进行排序。要按降序对值进行排序，可以使用`“ascending”`参数并将其设置为`False`。然后，`sort_values()` 方法将按降序对数据帧进行排序。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by="Marks",inplace=True,ignore_index=True,ascending=False)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya     85     A
1       1    12       Chris     95     A
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
4       1    15       Harry     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
7       2    15        Golu     79     B
8       2    27       Harsh     55     C
9       2    23       Clara     78     B
10      3    33        Tina     82     A
11      3    34         Amy     88     A
12      3    15    Prashant     78     B
13      3    27      Aditya     55     C
14      3    23  Radheshyam     78     B
15      3    11       Bobby     50     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
0       1    12       Chris     95     A
1       3    34         Amy     88     A
2       1    11      Aditya     85     A
3       3    33        Tina     82     A
4       2    15        Golu     79     B
5       1    16      Aditya     78     B
6       2    23       Clara     78     B
7       3    15    Prashant     78     B
8       3    23  Radheshyam     78     B
9       1    14         Sam     75     B
10      2    22         Tom     73     B
11      2     1        Joel     68     B
12      1    15       Harry     55     C
13      2    27       Harsh     55     C
14      3    27      Aditya     55     C
15      3    11       Bobby     50     D
```

在本例中，我们已经将`ascending`参数设置为`False`。因此，数据帧的行按`Marks`降序排列。

如果我们按多列对数据帧进行排序，也可以按降序对其进行排序，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by=["Class","Marks"],inplace=True,ignore_index=True,ascending=False)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya     85     A
1       1    12       Chris     95     A
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
4       1    15       Harry     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
7       2    15        Golu     79     B
8       2    27       Harsh     55     C
9       2    23       Clara     78     B
10      3    33        Tina     82     A
11      3    34         Amy     88     A
12      3    15    Prashant     78     B
13      3    27      Aditya     55     C
14      3    23  Radheshyam     78     B
15      3    11       Bobby     50     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
0       3    34         Amy     88     A
1       3    33        Tina     82     A
2       3    15    Prashant     78     B
3       3    23  Radheshyam     78     B
4       3    27      Aditya     55     C
5       3    11       Bobby     50     D
6       2    15        Golu     79     B
7       2    23       Clara     78     B
8       2    22         Tom     73     B
9       2     1        Joel     68     B
10      2    27       Harsh     55     C
11      1    12       Chris     95     A
12      1    11      Aditya     85     A
13      1    16      Aditya     78     B
14      1    14         Sam     75     B
15      1    15       Harry     55     C
```

在上面的例子中，dataframe 首先由`Class`列按降序排序。如果这些行对于`Class`列具有相同的值，则这些行按照`Marks`降序排序。

当按多列对数据帧进行排序时，您可以将一列`True`和`False`值传递给`ascending`参数。这有助于我们按照一列升序和另一列降序对数据帧进行排序。例如，考虑下面的例子。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by=["Class","Marks"],inplace=True,ignore_index=True,ascending=[True,False])
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya     85     A
1       1    12       Chris     95     A
2       1    14         Sam     75     B
3       1    16      Aditya     78     B
4       1    15       Harry     55     C
5       2     1        Joel     68     B
6       2    22         Tom     73     B
7       2    15        Golu     79     B
8       2    27       Harsh     55     C
9       2    23       Clara     78     B
10      3    33        Tina     82     A
11      3    34         Amy     88     A
12      3    15    Prashant     78     B
13      3    27      Aditya     55     C
14      3    23  Radheshyam     78     B
15      3    11       Bobby     50     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
0       1    12       Chris     95     A
1       1    11      Aditya     85     A
2       1    16      Aditya     78     B
3       1    14         Sam     75     B
4       1    15       Harry     55     C
5       2    15        Golu     79     B
6       2    23       Clara     78     B
7       2    22         Tom     73     B
8       2     1        Joel     68     B
9       2    27       Harsh     55     C
10      3    34         Amy     88     A
11      3    33        Tina     82     A
12      3    15    Prashant     78     B
13      3    23  Radheshyam     78     B
14      3    27      Aditya     55     C
15      3    11       Bobby     50     D
```

在本例中，我们按照`Class`和`Marks`列对数据帧进行了排序。在`ascending`参数中，我们给出了列表`[True, False].`，因此，数据帧首先由`Class`列按升序排序。如果这些行对于`Class`列具有相同的值，则这些行按照`Marks`降序排序。

## 在 Python 中使用 NaN 值对数据帧进行排序

在 python pandas 中，`NaN`值被视为浮点数。当我们使用`sort_values()`方法对包含`NaN`值的数据帧的行进行排序时，包含`NaN`值的行被放置在数据帧的底部，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by="Marks",inplace=True,ignore_index=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
he input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya   85.0     A
1       1    12       Chris   95.0     A
2       1    14         Sam   75.0     B
3       1    16      Aditya   78.0     B
4       1    15       Harry    NaN     C
5       2     1        Joel   68.0     B
6       2    22         Tom   73.0     B
7       2    15        Golu   79.0     B
8       2    27       Harsh   55.0     C
9       2    23       Clara    NaN     B
10      3    33        Tina   82.0     A
11      3    34         Amy   88.0     A
12      3    15    Prashant    NaN     B
13      3    27      Aditya   55.0     C
14      3    23  Radheshyam   78.0     B
15      3    11       Bobby   50.0     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
0       3    11       Bobby   50.0     D
1       2    27       Harsh   55.0     C
2       3    27      Aditya   55.0     C
3       2     1        Joel   68.0     B
4       2    22         Tom   73.0     B
5       1    14         Sam   75.0     B
6       1    16      Aditya   78.0     B
7       3    23  Radheshyam   78.0     B
8       2    15        Golu   79.0     B
9       3    33        Tina   82.0     A
10      1    11      Aditya   85.0     A
11      3    34         Amy   88.0     A
12      1    12       Chris   95.0     A
13      1    15       Harry    NaN     C
14      2    23       Clara    NaN     B
15      3    15    Prashant    NaN     B
```

在这个例子中，您可以观察到`Marks`列包含一些`NaN`值。当我们按照`Marks`列对数据帧进行排序时，在`Marks`列中具有`NaN`值的行被放置在排序后的数据帧的底部。

如果您想将带有`NaN`值的行放在数据帧的顶部，您可以在如下所示的`sort_values()`函数中将`na_position`参数设置为`“first”` 。

```py
import pandas as pd
grades=pd.read_csv("grade.csv")
print("The input dataframe is")
print(grades)
grades.sort_values(by="Marks",inplace=True,ignore_index=True,na_position="first")
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
    Class  Roll        Name  Marks Grade
0       1    11      Aditya   85.0     A
1       1    12       Chris   95.0     A
2       1    14         Sam   75.0     B
3       1    16      Aditya   78.0     B
4       1    15       Harry    NaN     C
5       2     1        Joel   68.0     B
6       2    22         Tom   73.0     B
7       2    15        Golu   79.0     B
8       2    27       Harsh   55.0     C
9       2    23       Clara    NaN     B
10      3    33        Tina   82.0     A
11      3    34         Amy   88.0     A
12      3    15    Prashant    NaN     B
13      3    27      Aditya   55.0     C
14      3    23  Radheshyam   78.0     B
15      3    11       Bobby   50.0     D
The sorted dataframe is
    Class  Roll        Name  Marks Grade
0       1    15       Harry    NaN     C
1       2    23       Clara    NaN     B
2       3    15    Prashant    NaN     B
3       3    11       Bobby   50.0     D
4       2    27       Harsh   55.0     C
5       3    27      Aditya   55.0     C
6       2     1        Joel   68.0     B
7       2    22         Tom   73.0     B
8       1    14         Sam   75.0     B
9       1    16      Aditya   78.0     B
10      3    23  Radheshyam   78.0     B
11      2    15        Golu   79.0     B
12      3    33        Tina   82.0     A
13      1    11      Aditya   85.0     A
14      3    34         Amy   88.0     A
15      1    12       Chris   95.0     A
```

在上面的例子中，我们已经将参数`na_position`设置为`"top"`。因此，`Marks`列具有`NaN`值的行被放置在排序后的数据帧的顶部。

## Python 中按行对数据帧的列进行排序

我们还可以根据行中的值对数据帧的列进行排序。为此，我们可以使用`sort_values()`函数中的`axis`参数。为了按行对数据帧的列进行排序，我们将把行的索引作为输入参数传递给`“by”`方法。此外，我们将在`sort_values()`方法中将`axis`参数设置为 1。执行后，`sort_values()` 方法将返回一个 dataframe，其中的列按给定的行排序。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("StudentMarks.csv",index_col="Student")
print("The input dataframe is")
print(grades)
grades.sort_values(by="Aditya",axis=1,inplace=True,ignore_index=True,na_position="first")
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
         Physics  Chemistry  Math  Biology  Arts
Student                                         
Aditya        92         76    95       73    91
Chris         95         96    79       71    93
Sam           65         62    75       95    63
Harry         68         92    69       66    98
Golu          74         95    96       76    64
Joel          99         79    77       91    61
Tom           72         94    61       65    69
Harsh         98         99    93       95    91
Clara         93         67    78       79    71
Tina          99         76    78       94    95
The sorted dataframe is
          0   1   2   3   4
Student                    
Aditya   73  76  91  92  95
Chris    71  96  93  95  79
Sam      95  62  63  65  75
Harry    66  92  98  68  69
Golu     76  95  64  74  96
Joel     91  79  61  99  77
Tom      65  94  69  72  61
Harsh    95  99  91  98  93
Clara    79  67  71  93  78
Tina     94  76  95  99  78
```

在上面的例子中，我们已经根据索引为`"Aditya"`的行对 dataframe 的列进行了排序。为此，我们将`axis`参数设置为 1，并将索引名传递给`sort_values()`方法的`by`参数。

在上面的输出数据帧中，您可以看到列名已经被删除。这是由于我们将`ignore_index`参数设置为`True`的原因。

如果您想保留列名，您可以删除参数`ignore_index`或将其设置为`False`，如下所示。

```py
import pandas as pd
grades=pd.read_csv("StudentMarks.csv",index_col="Student")
print("The input dataframe is")
print(grades)
grades.sort_values(by="Aditya",axis=1,inplace=True,na_position="first")
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
         Physics  Chemistry  Math  Biology  Arts
Student                                         
Aditya        92         76    95       73    91
Chris         95         96    79       71    93
Sam           65         62    75       95    63
Harry         68         92    69       66    98
Golu          74         95    96       76    64
Joel          99         79    77       91    61
Tom           72         94    61       65    69
Harsh         98         99    93       95    91
Clara         93         67    78       79    71
Tina          99         76    78       94    95
The sorted dataframe is
         Biology  Chemistry  Arts  Physics  Math
Student                                         
Aditya        73         76    91       92    95
Chris         71         96    93       95    79
Sam           95         62    63       65    75
Harry         66         92    98       68    69
Golu          76         95    64       74    96
Joel          91         79    61       99    77
Tom           65         94    69       72    61
Harsh         95         99    91       98    93
Clara         79         67    71       93    78
Tina          94         76    95       99    78
```

在本例中，您可以看到我们保留了数据帧的列名。这是因为我们移除了`ignore_index`参数，并将其设置为默认值`False`。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能还会喜欢这篇关于用 Python](https://codinginfinite.com/regression-in-machine-learning-with-examples/) 对混合数据类型进行聚类的[的文章。](https://codinginfinite.com/clustering-for-mixed-data-types-in-python/)

## sort index()方法

`sort_index()` 方法用于通过索引对熊猫数据帧进行排序。它具有以下语法。

```py
DataFrame.sort_index(*, axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)
```

*   `axis`参数用于决定我们是否要对数据帧的行或列进行排序。为了根据一列或一组列对数据帧中的行进行排序，我们可以将值 0 传递给参数`axis`，这是它的默认值。为了根据一行或多行对数据帧的列进行排序，我们可以将值 1 传递给`axis`参数。
*   `level`参数用于决定数据帧排序的索引级别。它有默认值`None`,表示所有索引级别都进行排序。如果希望按特定的索引级别对数据帧进行排序，可以将索引级别或索引名称传递给 level 参数。要按多个索引对数据帧进行排序，可以给`level`参数一个索引名或索引级别的列表。
*   `ascending`参数决定数据帧是按升序还是降序排序。默认情况下，`True`表示按升序排序。您可以将它设置为`False`，以降序对数据帧进行排序。对于具有多级索引的数据帧，您可以传递一个由`True`和`False`值组成的列表，以决定您希望哪个级别按升序排列，哪个级别按降序排列。
*   `inplace`参数用于决定我们是修改原始数据帧还是在排序后创建新的数据帧。默认情况下，`inplace`设置为`False`。因此，`sort_index()` 方法不会修改原始数据帧，而是返回新排序的数据帧。如果您想在排序时修改原始数据帧，可以将`inplace`设置为`True`。
*   `kind`参数用于决定排序算法。默认情况下，`sort_values()`方法使用快速排序算法。经过数据分析，如果你认为输入的数据有一个确定的模式，某种排序算法可以减少时间，你可以使用'`mergesort’, ‘heapsort’,` 或 `‘stable’`排序算法。
*   `na_position`参数用于决定具有`NaN`值的行的位置。默认情况下，它的值为`'last'`,表示具有`NaN`值的行最终存储在排序后的数据帧中。如果您想让带有`NaN`值的行位于排序数据帧的顶部，您可以将它设置为`“first”`。
*   `sort_remaining`参数用于具有多级索引的数据帧。如果您想按未在`level`参数中指定的级别对数据帧进行排序，您可以将`sort_remaining`参数设置为`True`。如果您不想按剩余的索引对数据帧进行排序，您可以将`sort_remaining`设置为`False`。
*   `key`参数用于在排序前对数据帧的索引执行操作。它接受一个矢量化函数作为其输入参数。提供给 key 参数的函数必须将一个 Index 对象作为其输入参数，并在执行后返回一个 Index 对象。在排序之前，该函数独立应用于输入数据帧中的每个索引列。

执行后，如果`inplace`参数设置为`False`，则`sort_index()` 方法返回排序后的数据帧。如果`inplace`被设置为`True`，则`sort_index()`方法返回`None`。

## 按索引排序熊猫数据帧

要按索引对 pandas 数据帧进行排序，可以在数据帧上使用`sort_index()`方法。为此，我们首先需要[创建一个带有索引](https://www.pythonforbeginners.com/basics/pandas-dataframe-index-in-python)的数据帧。然后，我们可以调用 dataframe 上的`sort_index()` 方法。执行后，`sort_index()`方法返回一个排序后的数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col="Roll")
print("The input dataframe is")
print(grades)
sorted_df=grades.sort_index()
print("The sorted dataframe is")
print(sorted_df)
```

输出:

```py
The input dataframe is
      Class        Name  Marks Grade
Roll                                
11        1      Aditya   85.0     A
12        1       Chris   95.0     A
14        1         Sam   75.0     B
16        1      Aditya   78.0     B
15        1       Harry    NaN     C
1         2        Joel   68.0     B
22        2         Tom   73.0     B
15        2        Golu   79.0     B
27        2       Harsh   55.0     C
23        2       Clara    NaN     B
33        3        Tina   82.0     A
34        3         Amy   88.0     A
15        3    Prashant    NaN     B
27        3      Aditya   55.0     C
23        3  Radheshyam   78.0     B
11        3       Bobby   50.0     D
The sorted dataframe is
      Class        Name  Marks Grade
Roll                                
1         2        Joel   68.0     B
11        1      Aditya   85.0     A
11        3       Bobby   50.0     D
12        1       Chris   95.0     A
14        1         Sam   75.0     B
15        1       Harry    NaN     C
15        2        Golu   79.0     B
15        3    Prashant    NaN     B
16        1      Aditya   78.0     B
22        2         Tom   73.0     B
23        2       Clara    NaN     B
23        3  Radheshyam   78.0     B
27        2       Harsh   55.0     C
27        3      Aditya   55.0     C
33        3        Tina   82.0     A
34        3         Amy   88.0     A
```

在上面的例子中，我们首先使用`read_csv()`方法读取一个 CSV 文件。在`read_csv()` 方法中，我们使用了`index_col`参数来指定`"Roll"` 列应该被用作数据帧的索引。当我们对由`read_csv()`方法返回的 dataframe 调用`sort_index()`方法时，它返回一个按索引列排序的 dataframe。

在上面的例子中，原始数据帧没有被修改。如果想修改原始数据帧，可以使用`sort_index()`方法中的`inplace=True`参数。执行后，原始数据帧将被修改。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col="Roll")
print("The input dataframe is")
print(grades)
grades.sort_index(inplace=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
      Class        Name  Marks Grade
Roll                                
11        1      Aditya   85.0     A
12        1       Chris   95.0     A
14        1         Sam   75.0     B
16        1      Aditya   78.0     B
15        1       Harry    NaN     C
1         2        Joel   68.0     B
22        2         Tom   73.0     B
15        2        Golu   79.0     B
27        2       Harsh   55.0     C
23        2       Clara    NaN     B
33        3        Tina   82.0     A
34        3         Amy   88.0     A
15        3    Prashant    NaN     B
27        3      Aditya   55.0     C
23        3  Radheshyam   78.0     B
11        3       Bobby   50.0     D
The sorted dataframe is
      Class        Name  Marks Grade
Roll                                
1         2        Joel   68.0     B
11        1      Aditya   85.0     A
11        3       Bobby   50.0     D
12        1       Chris   95.0     A
14        1         Sam   75.0     B
15        1       Harry    NaN     C
15        2        Golu   79.0     B
15        3    Prashant    NaN     B
16        1      Aditya   78.0     B
22        2         Tom   73.0     B
23        2       Clara    NaN     B
23        3  Radheshyam   78.0     B
27        2       Harsh   55.0     C
27        3      Aditya   55.0     C
33        3        Tina   82.0     A
34        3         Amy   88.0     A
```

在本例中，您可以看到原始数据帧已经排序。这是因为我们在`sort_index()`方法中将`inplace`参数设置为`True`。

如果数据帧中有多级索引，并且希望按特定索引对数据帧进行排序，可以将索引级别传递给`sort_index()`方法中的`level`参数。

在下面的例子中，`Class`和`Roll`列都被用作索引。`Class`列用作主索引，而`Roll`列用作辅助索引。要仅通过`Roll`列对数据帧进行排序，我们将使用`level`参数并将其设置为 1。这样，输入数据帧将按`Roll`列排序。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col=["Class","Roll"])
print("The input dataframe is")
print(grades)
grades.sort_index(level=1,inplace=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      16        Aditya   78.0     B
      15         Harry    NaN     C
2     1           Joel   68.0     B
      22           Tom   73.0     B
      15          Golu   79.0     B
      27         Harsh   55.0     C
      23         Clara    NaN     B
3     33          Tina   82.0     A
      34           Amy   88.0     A
      15      Prashant    NaN     B
      27        Aditya   55.0     C
      23    Radheshyam   78.0     B
      11         Bobby   50.0     D
The sorted dataframe is
                  Name  Marks Grade
Class Roll                         
2     1           Joel   68.0     B
1     11        Aditya   85.0     A
3     11         Bobby   50.0     D
1     12         Chris   95.0     A
      14           Sam   75.0     B
      15         Harry    NaN     C
2     15          Golu   79.0     B
3     15      Prashant    NaN     B
1     16        Aditya   78.0     B
2     22           Tom   73.0     B
      23         Clara    NaN     B
3     23    Radheshyam   78.0     B
2     27         Harsh   55.0     C
3     27        Aditya   55.0     C
      33          Tina   82.0     A
      34           Amy   88.0     A
```

在上面的例子中，我们使用索引级别作为`level`参数的输入参数。或者，您也可以将索引级别的名称传递给参数`level`，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col=["Class","Roll"])
print("The input dataframe is")
print(grades)
grades.sort_index(level="Roll",inplace=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      16        Aditya   78.0     B
      15         Harry    NaN     C
2     1           Joel   68.0     B
      22           Tom   73.0     B
      15          Golu   79.0     B
      27         Harsh   55.0     C
      23         Clara    NaN     B
3     33          Tina   82.0     A
      34           Amy   88.0     A
      15      Prashant    NaN     B
      27        Aditya   55.0     C
      23    Radheshyam   78.0     B
      11         Bobby   50.0     D
The sorted dataframe is
                  Name  Marks Grade
Class Roll                         
2     1           Joel   68.0     B
1     11        Aditya   85.0     A
3     11         Bobby   50.0     D
1     12         Chris   95.0     A
      14           Sam   75.0     B
      15         Harry    NaN     C
2     15          Golu   79.0     B
3     15      Prashant    NaN     B
1     16        Aditya   78.0     B
2     22           Tom   73.0     B
      23         Clara    NaN     B
3     23    Radheshyam   78.0     B
2     27         Harsh   55.0     C
3     27        Aditya   55.0     C
      33          Tina   82.0     A
      34           Amy   88.0     A
```

在上面的例子中，我们使用参数`level="Roll"`而不是`level=1` 对输入数据帧进行排序。在这两种情况下，输出是相同的。

按照指定的索引排序后，`sort_index()` 方法还按照剩余的索引对数据帧进行排序。要停止这种情况，您可以将`sort_remaining`参数设置为`False`，如下例所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col=["Class","Roll"])
print("The input dataframe is")
print(grades)
grades.sort_index(level="Roll",inplace=True,sort_remaining=False)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      16        Aditya   78.0     B
      15         Harry    NaN     C
2     1           Joel   68.0     B
      22           Tom   73.0     B
      15          Golu   79.0     B
      27         Harsh   55.0     C
      23         Clara    NaN     B
3     33          Tina   82.0     A
      34           Amy   88.0     A
      15      Prashant    NaN     B
      27        Aditya   55.0     C
      23    Radheshyam   78.0     B
      11         Bobby   50.0     D
The sorted dataframe is
                  Name  Marks Grade
Class Roll                         
2     1           Joel   68.0     B
1     11        Aditya   85.0     A
3     11         Bobby   50.0     D
1     12         Chris   95.0     A
      14           Sam   75.0     B
      15         Harry    NaN     C
2     15          Golu   79.0     B
3     15      Prashant    NaN     B
1     16        Aditya   78.0     B
2     22           Tom   73.0     B
      23         Clara    NaN     B
3     23    Radheshyam   78.0     B
2     27         Harsh   55.0     C
3     27        Aditya   55.0     C
      33          Tina   82.0     A
      34           Amy   88.0     A
```

在上面的例子中，如果两行在`"Roll"`列中有相同的值，并且`sort_remaining`参数没有设置为`False`，那么`sort_index()`方法将根据`Class`索引对数据帧进行排序。为了阻止`sort_index()`方法这样做，我们使用了`sort_remaining`参数并将其设置为`False`。

## 用 Python 对熊猫数据帧进行多索引排序

要通过多个索引对 pandas 数据帧进行排序，可以将索引级别列表传递给`sort_index()`方法的`level`参数，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col=["Class","Roll"])
print("The input dataframe is")
print(grades)
grades.sort_index(level=[0,1],inplace=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      16        Aditya   78.0     B
      15         Harry    NaN     C
2     1           Joel   68.0     B
      22           Tom   73.0     B
      15          Golu   79.0     B
      27         Harsh   55.0     C
      23         Clara    NaN     B
3     33          Tina   82.0     A
      34           Amy   88.0     A
      15      Prashant    NaN     B
      27        Aditya   55.0     C
      23    Radheshyam   78.0     B
      11         Bobby   50.0     D
The sorted dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      15         Harry    NaN     C
      16        Aditya   78.0     B
2     1           Joel   68.0     B
      15          Golu   79.0     B
      22           Tom   73.0     B
      23         Clara    NaN     B
      27         Harsh   55.0     C
3     11         Bobby   50.0     D
      15      Prashant    NaN     B
      23    Radheshyam   78.0     B
      27        Aditya   55.0     C
      33          Tina   82.0     A
      34           Amy   88.0     A
```

在上面的例子中，我们将`Class`和`Roll`列作为索引。当我们通过`level=[0,1]`时，`sort_index()`方法首先通过`Class`列对输入数据帧进行排序。如果两行的`Class`列具有相同的值，它将根据`Roll`列对它们进行排序。

除了索引级别，您还可以将索引级别的名称传递给参数`level`，如下例所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col=["Class","Roll"])
print("The input dataframe is")
print(grades)
grades.sort_index(level=["Class","Roll"],inplace=True)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      16        Aditya   78.0     B
      15         Harry    NaN     C
2     1           Joel   68.0     B
      22           Tom   73.0     B
      15          Golu   79.0     B
      27         Harsh   55.0     C
      23         Clara    NaN     B
3     33          Tina   82.0     A
      34           Amy   88.0     A
      15      Prashant    NaN     B
      27        Aditya   55.0     C
      23    Radheshyam   78.0     B
      11         Bobby   50.0     D
The sorted dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      15         Harry    NaN     C
      16        Aditya   78.0     B
2     1           Joel   68.0     B
      15          Golu   79.0     B
      22           Tom   73.0     B
      23         Clara    NaN     B
      27         Harsh   55.0     C
3     11         Bobby   50.0     D
      15      Prashant    NaN     B
      23    Radheshyam   78.0     B
      27        Aditya   55.0     C
      33          Tina   82.0     A
      34           Amy   88.0     A
```

在上面的例子中，我们使用参数`level=["Class", "Roll"]`而不是`level=[0, 1]`对输入数据帧进行排序。在这两种情况下，输出是相同的。

## 按索引降序排列熊猫数据帧

要按索引降序排列数据帧，可以将`sort_index()` 方法中的`ascending`参数设置为 False，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col=["Class","Roll"])
print("The input dataframe is")
print(grades)
grades.sort_index(level="Roll",inplace=True,ascending=False)
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      16        Aditya   78.0     B
      15         Harry    NaN     C
2     1           Joel   68.0     B
      22           Tom   73.0     B
      15          Golu   79.0     B
      27         Harsh   55.0     C
      23         Clara    NaN     B
3     33          Tina   82.0     A
      34           Amy   88.0     A
      15      Prashant    NaN     B
      27        Aditya   55.0     C
      23    Radheshyam   78.0     B
      11         Bobby   50.0     D
The sorted dataframe is
                  Name  Marks Grade
Class Roll                         
3     34           Amy   88.0     A
      33          Tina   82.0     A
      27        Aditya   55.0     C
2     27         Harsh   55.0     C
3     23    Radheshyam   78.0     B
2     23         Clara    NaN     B
      22           Tom   73.0     B
1     16        Aditya   78.0     B
3     15      Prashant    NaN     B
2     15          Golu   79.0     B
1     15         Harry    NaN     C
      14           Sam   75.0     B
      12         Chris   95.0     A
3     11         Bobby   50.0     D
1     11        Aditya   85.0     A
2     1           Joel   68.0     B
```

在上面的例子中，`sort_index()`方法按照`Class`和`Roll`列以降序对输入数据帧进行排序。

当通过多个索引对数据帧进行排序时，您可以将一列`True`和`False`值传递给`ascending`参数，如下所示。

```py
import pandas as pd
grades=pd.read_csv("grade.csv",index_col=["Class","Roll"])
print("The input dataframe is")
print(grades)
grades.sort_index(level=["Class","Roll"],inplace=True,ascending=[False,True])
print("The sorted dataframe is")
print(grades)
```

输出:

```py
The input dataframe is
                  Name  Marks Grade
Class Roll                         
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      16        Aditya   78.0     B
      15         Harry    NaN     C
2     1           Joel   68.0     B
      22           Tom   73.0     B
      15          Golu   79.0     B
      27         Harsh   55.0     C
      23         Clara    NaN     B
3     33          Tina   82.0     A
      34           Amy   88.0     A
      15      Prashant    NaN     B
      27        Aditya   55.0     C
      23    Radheshyam   78.0     B
      11         Bobby   50.0     D
The sorted dataframe is
                  Name  Marks Grade
Class Roll                         
3     11         Bobby   50.0     D
      15      Prashant    NaN     B
      23    Radheshyam   78.0     B
      27        Aditya   55.0     C
      33          Tina   82.0     A
      34           Amy   88.0     A
2     1           Joel   68.0     B
      15          Golu   79.0     B
      22           Tom   73.0     B
      23         Clara    NaN     B
      27         Harsh   55.0     C
1     11        Aditya   85.0     A
      12         Chris   95.0     A
      14           Sam   75.0     B
      15         Harry    NaN     C
      16        Aditya   78.0     B
```

在本例中，我们按照`Class`和`Marks`列对数据帧进行了排序。在`ascending`参数中，我们已经给出了列表 `[False, True]`。因此，数据帧首先由`Class`列按`descending`顺序排序。如果这些行对于`Class`列具有相同的值，则这些行按`Marks`升序排序。

## 结论

在本文中，我们讨论了使用`sort_values()` 和 `the sort_index()` 方法在 Python 中对熊猫数据帧进行排序的不同方法。

要了解更多关于 python 编程的知识，你可以阅读这篇关于 python 中的[字典理解的文章。你可能也会喜欢这篇关于 python](https://www.pythonforbeginners.com/dictionary/dictionary-comprehension-in-python) 中的[列表理解的文章。](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)