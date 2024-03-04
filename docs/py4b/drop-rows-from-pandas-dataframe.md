# 从 Pandas 数据框架中删除行

> 原文：<https://www.pythonforbeginners.com/basics/drop-rows-from-pandas-dataframe>

在 Python 中，我们使用 pandas 数据帧来完成许多数据处理任务。有时，由于各种原因，我们需要从数据帧中删除一些行。在本文中，我们将讨论使用`drop()`方法从 [pandas 数据帧](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)中删除行的不同方法。

## drop()方法

d `rop()`方法可以用来从 pandas 数据帧中删除列或行。它具有以下语法。

```py
DataFrame.drop(labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
```

这里，

*   当我们必须从数据帧中删除一行时，使用`index`参数。`index`参数将必须删除的一个索引或索引列表作为其输入参数。
*   当我们需要从数据帧中删除一列时，使用`columns`参数。`columns`参数将需要删除的列名或列名列表作为其输入参数。
*   `labels`参数表示我们需要从数据帧中移除的索引或列标签。为了从数据帧中删除行，我们使用索引标签。为了删除两行或更多行，我们还可以向参数`labels`传递一个索引列表。
*   当我们不使用`index`参数时，我们可以将需要删除的行的索引传递给`labels`参数作为它的输入参数。在这种情况下，我们使用`axis`参数来决定是删除一行还是一列。如果我们想从数据帧中删除一列，我们将`axis`参数设置为 1。当我们想从数据帧中删除一行时，我们将`axis`参数设置为 0，这是它的默认值。
*   当我们有多级索引时，`level`参数用于从数据帧中删除行。`level`参数接受我们要从数据帧中删除的行的索引级别或索引名称。要删除两个或更多级别，可以将索引级别或索引名称的列表传递给`level`参数。
*   `inplace`参数用于决定我们是在删除操作后获得一个新的数据帧，还是想要修改原始数据帧。当`inplace`被设置为 False(这是它的默认值)时，原始的数据帧不变，并且`drop()`方法在执行后返回修改后的数据帧。要修改原始数据帧，您可以将`inplace`设置为 True。
*   `errors`参数用于决定我们是否希望在执行`drop()`方法时引发异常和错误。默认情况下，`errors`参数设置为`“raise”`。因此，如果执行过程中出现任何问题，`drop()`方法就会引发异常。如果不希望出现错误，可以将错误参数设置为`“ignore”`。在这之后，`drop()`方法将抑制所有的异常。

执行后，如果`inplace`参数设置为 False， `drop()`方法返回一个新的数据帧。否则，它修改原始数据帧并返回`None`。

## 通过索引标签从 Pandas 数据帧中删除行

为了通过索引标签删除数据帧的列，我们将把索引标签传递给`drop()`方法中的`labels`参数。执行后，`drop()`方法将返回一个数据帧，其中包含除了带有在`labels`参数中指定的索引标签的行之外的所有行。您可以在下面的示例中观察到这一点。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
print("After dropping rows with index 55")
df=df.drop(labels=55)
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping rows with index 55
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
78         3    23  Radheshyam     B
50         3    11       Bobby     D
```

在上面的例子中，我们已经使用 csv 文件创建了一个数据帧。然后，我们删除了数据帧中索引为 55 的行。在输出数据帧中，您可以观察到索引为 55 的所有行都不存在。因此，`drop()`方法删除了具有指定索引的行。

除了使用`labels`参数，您还可以在`drop()`方法中使用`index`参数从数据帧中删除一行，如下例所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
print("After dropping rows with index 55")
df=df.drop(index=55)
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping rows with index 55
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
78         3    23  Radheshyam     B
50         3    11       Bobby     D
```

在上面的例子中，我们使用了`index`参数而不是`labels`参数来传递索引值作为`drop()`方法的输入。您可以观察到两种情况下的输出是相同的。因此，您可以使用任何一个`index`或`labels`参数从 pandas 数据帧中删除行。

## 按位置从 Pandas 数据帧中删除行

要按位置从数据帧中删除行，我们将使用以下步骤。

*   首先，我们将使用`index`属性获取数据帧的索引对象。
*   接下来，我们将使用 indexing 操作符获取索引对象的元素，该元素位于我们要从数据帧中删除的行的位置。这个元素将是我们要删除的行的标签。
*   在获得要删除的行的标签后，我们可以将标签传递给`labels`参数，作为`drop()`方法中的输入参数。

在执行了`drop()`方法之后，我们将得到如下所示的修改后的数据帧。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
position=3
print("After dropping row at position 3")
idx=df.index[position-1]
df=df.drop(labels=idx)
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping row at position 3
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
```

在上面的例子中，您可以看到我们在数据帧的第三个位置删除了该行。这里，第三个位置的行的索引为 82。因此，如果存在索引为 82 的任何其他行，该行也将从输入数据帧中删除。

在上面的例子中，还可以将从 index 对象获得的索引标签传递给`drop()` 方法中的`index`参数。程序执行后你会得到同样的结果。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
position=3
print("After dropping row at position 3")
idx=df.index[position-1]
df=df.drop(index=idx)
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping row at position 3
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
```

## 从熊猫数据框中删除第一行

要从数据帧中删除第一行，我们将首先使用 index 属性获取第一行的索引标签。

然后，我们将索引标签传递给`drop()` 方法中的`labels`参数，以删除 dataframe 的第一行，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
position=1
print("After dropping first row")
idx=df.index[position-1]
df=df.drop(index=idx)
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping first row
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
78         3    23  Radheshyam     B
50         3    11       Bobby     D
```

在本例中，我们首先使用[数据帧索引](https://www.pythonforbeginners.com/basics/pandas-dataframe-index-in-python)和索引操作符来获得第一个位置的行的索引标签，即索引 55。然后，我们将索引标签传递给了`drop()`方法中的`index`参数。

在输出中，您可以观察到从数据帧中删除了不止一行。这是因为`drop()`方法通过索引标签来删除行。因此，与第一行具有相同索引的所有行都将从输入数据帧中删除。

## 从熊猫数据帧中删除最后一行

为了从数据帧中删除最后一行，我们将首先使用`len()`函数获得数据帧中的总行数。`len()`函数将 dataframe 作为其输入参数，并返回 dataframe 中的总行数。

在获得总行数之后，我们将使用`index`属性获得最后一行的索引标签。此后，我们将把索引标签传递给`drop()` 方法中的`labels`参数，以删除 dataframe 的最后一行，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
total_rows=len(df)
position=total_rows-1
print("After dropping last row")
idx=df.index[position]
df=df.drop(labels=idx)
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping last row
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
```

在本例中，我们删除了输入数据帧的最后一行。同样，如果输入数据帧包含与最后一行具有相同索引的行，所有这样的行也将被删除。

## 将行放到数据帧中的适当位置

在前面几节给出的示例中，您可以观察到原始数据帧在删除行后没有被修改。相反，一个新的 dataframe 被创建并由`drop()`方法返回。如果您想修改现有的数据帧而不是创建一个新的数据帧，您可以在如下所示的`drop()` 方法中将`inplace`参数设置为 True。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
total_rows=len(df)
position=total_rows-1
print("After dropping last row")
idx=df.index[position]
df.drop(index=idx,inplace=True)
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping last row
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
```

在这个例子中，我们已经在`drop()` 方法中将`inplace`参数设置为 True。因此，输入数据帧被修改，而不是创建新的数据帧。在这种情况下，`drop()`方法返回 None。

## 如果熊猫数据框架中存在索引，则删除行

如果传递给`drop()`方法的索引标签在 dataframe 中不存在，`drop()` 方法就会遇到如下所示的 [python KeyError](https://www.pythonforbeginners.com/basics/python-keyerror) 异常。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
print("After dropping row at index 1117")
df.drop(index=1117,inplace=True)
print("The modified dataframe is:")
print(df)
```

输出:

```py
KeyError: '[1117] not found in axis'
```

在上面的示例中，我们试图从输入数据帧中删除索引为 1117 的列。索引 1117 不存在于输入数据帧中。因此，`drop()`方法会遇到一个 KeyError 异常。

默认情况下，如果传递给`labels`或`index`参数的索引标签在 dataframe 中不存在，`drop()`方法会引发 KeyError 异常。为了[在索引不存在时抑制异常](https://www.pythonforbeginners.com/basics/suppress-exceptions-in-python)并在索引存在时删除行，您可以将`errors`参数设置为“`ignore”`，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
print("After dropping row at index 1117")
df.drop(index=1117,inplace=True,errors="ignore")
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping row at index 1117
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
```

在这个例子中，我们通过在`drop()`方法中将 errors 参数设置为`"ignore"`来抑制异常。因此，当传递给标签的索引标签或索引参数在输入数据帧中不存在时，`drop()` 方法对输入数据帧没有影响。

## 在 Pandas 数据帧中通过索引标签删除多行

要在 pandas 数据帧中通过索引标签删除多行，您可以将包含索引标签的列表传递给如下所示的`drop()` 方法。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
indices=[55,88]
print("After dropping rows at indices 55,88")
df.drop(index=indices,inplace=True,errors="ignore")
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping rows at indices 55,88
The modified dataframe is:
       Class  Roll        Name Grade
Marks                               
78         2    23       Clara     B
82         3    33        Tina     A
78         3    15    Prashant     B
78         3    23  Radheshyam     B
50         3    11       Bobby     D
```

在上面的例子中，我们已经将 list [55，88]传递给了`drop()` 方法中的`index`参数。因此，索引为 55 和 88 的所有行都将从输入数据帧中删除。

建议阅读:如果你对机器学习感兴趣，可以阅读这篇 [MLFlow 教程，里面有代码示例](https://codinginfinite.com/mlflow-tutorial-with-code-example/)。你可能也会喜欢这篇关于 2023 年 T2 15 款免费数据可视化工具的文章。

## 从 Pandas 数据框架中按位置删除多行

要从一个数据帧中按位置删除多行，我们将首先使用 [python 索引](https://avidpython.com/python-basics/python-indexing-operation-on-string-list-and-tuple/)和`index`属性找到我们想要删除的位置上所有行的索引标签。然后，我们将把索引标签列表传递给`drop()` 方法中的`labels`参数，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
positions=[3,4,5]
indices=[df.index[i-1] for i in positions]
print("After dropping rows at positions 3,4,5")
df.drop(labels=indices,inplace=True,errors="ignore")
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping rows at positions 3,4,5
The modified dataframe is:
       Class  Roll    Name Grade
Marks                           
55         2    27   Harsh     C
55         3    27  Aditya     C
50         3    11   Bobby     D
```

在上面的例子中，我们删除了位置 3、4 和 5 的行。为此，我们使用了[列表理解](https://www.pythonforbeginners.com/basics/list-comprehensions-in-python)和索引来获取指定位置的索引标签。然后，我们将索引列表传递给`drop()`方法中的`labels`参数，以便根据 pandas 数据帧中的位置删除行。

## 删除熊猫数据帧中的前 N 行

为了删除数据帧的前 n 行，我们将首先使用数据帧的`index`属性找到前 n 行的索引标签。然后，我们将把索引标签传递给`drop()`方法，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
n=3
indices=[df.index[i] for i in range(n)]
print("After dropping first 3 rows")
df.drop(index=indices,inplace=True,errors="ignore")
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping first 3 rows
The modified dataframe is:
       Class  Roll    Name Grade
Marks                           
88         3    34     Amy     A
50         3    11   Bobby     D
```

在上面的例子中，我们只删除了熊猫数据帧的前三行。但是，当执行`drop()`方法时，会删除更多的行。这是因为`drop()` 方法通过索引删除行。因此，与前三行具有相同索引的所有行将从数据帧中删除。

## 删除数据帧的最后 N 行

为了删除数据帧的最后 n 行，我们将首先使用`len()` 函数找到数据帧中的总行数。然后，我们将使用 index 属性和索引操作符找到最后 n 行的索引标签。获得索引标签后，我们将把它们传递给 `drop()`方法中的`labels`参数，以删除这些行，如下所示。

```py
import pandas as pd
df=pd.read_csv("grade2.csv",index_col="Marks")
print("The dataframe is:")
print(df)
total_rows=len(df)
n=3
indices=[df.index[i] for i in range(total_rows-n,total_rows)]
print("After dropping last 3 rows")
df.drop(index=indices,inplace=True,errors="ignore")
print("The modified dataframe is:")
print(df)
```

输出:

```py
The dataframe is:
       Class  Roll        Name Grade
Marks                               
55         2    27       Harsh     C
78         2    23       Clara     B
82         3    33        Tina     A
88         3    34         Amy     A
78         3    15    Prashant     B
55         3    27      Aditya     C
78         3    23  Radheshyam     B
50         3    11       Bobby     D
After dropping last 3 rows
The modified dataframe is:
       Class  Roll  Name Grade
Marks                         
82         3    33  Tina     A
88         3    34   Amy     A
```

在这个例子中，我们只删除了熊猫数据帧的最后三行。但是，当执行`drop()`方法时，会删除更多的行。这是因为`drop()` 方法通过索引删除行。因此，将从数据帧中删除与最后三行具有相同索引的所有行。

## 结论

在本文中，我们讨论了从 pandas 数据帧中删除行的不同方法。要了解更多关于 pandas 模块的信息，你可以阅读这篇关于如何对 pandas 数据帧进行排序的文章。你可能也会喜欢这篇关于如何从熊猫数据框中[删除列的文章。](https://www.pythonforbeginners.com/basics/drop-columns-from-pandas-dataframe)

我希望你喜欢阅读这篇文章。请继续关注更多内容丰富的文章。

快乐学习！