# 从 Pandas 数据框架中删除重复的行

> 原文：<https://www.pythonforbeginners.com/basics/drop-duplicate-rows-from-a-pandas-dataframe>

Pandas 数据帧用于在 Python 中处理表格数据。数据有时包含可能不需要的重复值。在本文中，我们将讨论使用 `drop_duplicates()`方法从 pandas 数据帧中删除重复行的不同方法。

## drop_duplicates 方法

方法用于从 pandas 数据帧中删除重复的行。它具有以下语法。

```py
DataFrame.drop_duplicates(subset=None, *, keep='first', inplace=False, ignore_index=False)
```

这里，

*   `subset`参数用于比较两行以确定重复行。默认情况下，`subset`参数设置为无。因此，所有列中的值都用于行中的比较。如果您想只通过一列来比较两行，您可以将列名作为输入参数传递给`subset`参数。如果您想通过两列或更多列来比较行，您可以将列名列表传递给`subset`参数。
*   `keep`参数用于决定我们是否希望在输出数据帧中保留一个重复行。如果我们想删除除第一次出现之外的所有重复行，我们可以将参数`keep`设置为默认值`“first”`。如果我们想删除除最后一个重复行之外的所有重复行，我们可以将`keep`参数设置为`“last”`。如果我们需要删除所有重复的行，我们可以将`keep`参数设置为 False。
*   `inplace`参数用于决定我们是在删除操作后获得一个新的数据帧，还是想要修改原始数据帧。当 inplace 设置为 False(这是其默认值)时，原始数据帧不会更改，drop_duplicates()方法会在执行后返回修改后的数据帧。要改变原始数据帧，可以将 inplace 设置为 True。
*   当从数据帧中删除行时，索引的顺序变得不规则。如果要刷新索引，将有序索引从 0 分配到`(length of dataframe)-1`，可以将`ignore_index`设置为 True。

执行后，如果`inplace`参数设置为 False，`drop_duplicates()`方法返回一个数据帧。否则，它返回 None。

## 从 Pandas 数据框架中删除重复的行

要从 pandas 数据帧中删除重复的行，可以在数据帧上调用`drop_duplicates()`方法。执行后，它返回一个包含所有唯一行的数据帧。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df=df.drop_duplicates()
print("After dropping duplicates:")
print(df)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
0       2    27       Harsh     55     C
1       2    23       Clara     78     B
2       3    33        Tina     82     A
3       3    34         Amy     88     A
4       3    15    Prashant     78     B
5       3    27      Aditya     55     C
6       3    34         Amy     88     A
7       3    23  Radheshyam     78     B
8       3    11       Bobby     50     D
9       2    27       Harsh     55     C
10      3    15      Lokesh     88     A
After dropping duplicates:
    Class  Roll        Name  Marks Grade
0       2    27       Harsh     55     C
1       2    23       Clara     78     B
2       3    33        Tina     82     A
3       3    34         Amy     88     A
4       3    15    Prashant     78     B
5       3    27      Aditya     55     C
7       3    23  Radheshyam     78     B
8       3    11       Bobby     50     D
10      3    15      Lokesh     88     A
```

在上面的例子中，我们有一个输入数据帧，包含一些学生的班级、名单、姓名、分数和成绩。正如您所看到的，输入数据帧包含一些重复的行。索引 0 和 9 处的行是相同的。类似地，索引 3 和 6 处的行是相同的。在执行了`drop_duplicates()`方法之后，我们得到了一个[熊猫数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)，其中所有的行都是唯一的。因此，从数据帧中删除索引 6 和 9 处的行，以便索引 0 和 3 处的行变得唯一。

## 从 Pandas 数据框架中删除所有重复的行

在上面的示例中，保留了每组重复行中的一个条目。如果想从 dataframe 中删除所有重复的行，可以在`drop_duplicates()` 方法中将`keep`参数设置为 False。此后，所有具有重复值的行将被删除。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df=df.drop_duplicates(keep=False)
print("After dropping duplicates:")
print(df)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
0       2    27       Harsh     55     C
1       2    23       Clara     78     B
2       3    33        Tina     82     A
3       3    34         Amy     88     A
4       3    15    Prashant     78     B
5       3    27      Aditya     55     C
6       3    34         Amy     88     A
7       3    23  Radheshyam     78     B
8       3    11       Bobby     50     D
9       2    27       Harsh     55     C
10      3    15      Lokesh     88     A
After dropping duplicates:
    Class  Roll        Name  Marks Grade
1       2    23       Clara     78     B
2       3    33        Tina     82     A
4       3    15    Prashant     78     B
5       3    27      Aditya     55     C
7       3    23  Radheshyam     78     B
8       3    11       Bobby     50     D
10      3    15      Lokesh     88     A
```

在本例中，您可以观察到索引 0 和 9 处的行是相同的。类似地，索引 3 和 6 处的行是相同的。当我们在`drop_duplicates()`方法中将`keep`参数设置为 False 时，您可以观察到所有具有重复值的行(即索引为 0、3、6 和 9 的行)都从输入数据帧中删除了。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇 [MLFlow 教程，里面有代码示例](https://codinginfinite.com/mlflow-tutorial-with-code-example/)。你可能也会喜欢这篇关于 2023 年 T2 15 款免费数据可视化工具的文章。

## 从 Pandas 数据帧中删除重复的行

默认情况下，`drop_duplicates()` 方法返回一个新的数据帧。如果您想改变原始数据帧而不是创建一个新的，您可以在如下所示的`drop_duplicates()` 方法中将`inplace`参数设置为 True。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df.drop_duplicates(keep=False,inplace=True)
print("After dropping duplicates:")
print(df)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
0       2    27       Harsh     55     C
1       2    23       Clara     78     B
2       3    33        Tina     82     A
3       3    34         Amy     88     A
4       3    15    Prashant     78     B
5       3    27      Aditya     55     C
6       3    34         Amy     88     A
7       3    23  Radheshyam     78     B
8       3    11       Bobby     50     D
9       2    27       Harsh     55     C
10      3    15      Lokesh     88     A
After dropping duplicates:
    Class  Roll        Name  Marks Grade
1       2    23       Clara     78     B
2       3    33        Tina     82     A
4       3    15    Prashant     78     B
5       3    27      Aditya     55     C
7       3    23  Radheshyam     78     B
8       3    11       Bobby     50     D
10      3    15      Lokesh     88     A
```

在这个例子中，我们已经在 `drop_duplicates()` 方法中将`inplace`参数设置为 True。因此，`drop_duplicates()` 方法修改输入数据帧，而不是创建一个新的。这里，`drop_duplicates()`方法返回 None。

## 删除特定列中有重复值的行

默认情况下，`drop_duplicates()` 方法比较所有列的相似性，以检查重复的行。如果您想基于特定的列比较重复值的行，您可以使用`drop_duplicates()`方法中的`subset`参数。

`subset`参数将一列作为它的输入参数。此后，`drop_duplicates()` 方法只根据指定的列比较行。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv")
print("The dataframe is:")
print(df)
df.drop_duplicates(subset=["Class","Roll"],inplace=True)
print("After dropping duplicates:")
print(df)
```

输出:

```py
The dataframe is:
    Class  Roll        Name  Marks Grade
0       2    27       Harsh     55     C
1       2    23       Clara     78     B
2       3    33        Tina     82     A
3       3    34         Amy     88     A
4       3    15    Prashant     78     B
5       3    27      Aditya     55     C
6       3    34         Amy     88     A
7       3    23  Radheshyam     78     B
8       3    11       Bobby     50     D
9       2    27       Harsh     55     C
10      3    15      Lokesh     88     A
After dropping duplicates:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
7      3    23  Radheshyam     78     B
8      3    11       Bobby     50     D
```

在这个例子中，我们将 [python 列表](https://www.pythonforbeginners.com/lists/python-lists-cheat-sheet-2) ["Class "，" Roll"]传递给了`drop_duplicates()`方法中的`subset`参数。因此，仅根据这两列来确定重复行。因此，在`Class`和`Roll`列中具有相同值的行被认为是重复的，并从数据帧中删除。

## 结论

在本文中，我们讨论了使用`drop_duplicates()`方法从数据帧中删除重复行的不同方法。

要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！