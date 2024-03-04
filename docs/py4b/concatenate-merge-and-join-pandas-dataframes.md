# 连接、合并和联接熊猫数据框架

> 原文：<https://www.pythonforbeginners.com/basics/concatenate-merge-and-join-pandas-dataframes>

Pandas 数据框架是 python 中分析表格数据的主要工具。在本文中，我们将讨论使用`merge()` 函数、`join()`函数和`concat()`函数连接、合并和[联接熊猫数据帧](https://www.pythonforbeginners.com/basics/inner-join-dataframes-in-python)的不同方法。

## 熊猫 merge()函数

`merge()`函数用于在 Python 中合并熊猫数据帧。合并的发生方式类似于数据库列中的联接操作。`merge()`函数的语法如下。

```py
pandas.merge(left_df, right_df, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, suffixes=('_x', '_y'),indicator=False)
```

这里，

*   `left_df`和`right_df`参数各取一个数据帧作为它们的输入参数。数据帧的位置会影响输出。因此，我们明确地命名了数据帧的位置。
*   `“how”`参数用于决定将在`left_df`和`right_df`上执行的加入操作。它有默认值`“inner”`，表示将执行内部连接操作。您还可以使用文字`“outer”`、`“left”`或`“right”`分别在数据帧上执行完全外连接、左连接或右连接。
*   `on`参数用于决定用作连接操作的键的列。这里，提供给 on 参数的列名必须同时出现在`left_df`和`right_df`中。如果使用两个数据帧中不同的列名作为连接键，默认情况下 on 参数设置为`None`。
*   `left_on`参数用于指定用作来自`left_df`的连接键的列名。
*   `right_on`参数用于指定用作来自`right_df`的连接键的列名。
*   如果输入数据帧有任何列作为索引，您可以使用`left_index`和`right_index`参数来使用索引作为连接键。
*   如果您想使用`left_df`的索引作为连接键，那么`left_index`参数被设置为`True`。它有默认值`False`。
*   如果您想使用`right_df`的索引作为连接键，那么`right_index`参数被设置为`True`。它有默认值`False`。
*   当`left_df`和`right_df`有共同的列名时，使用 suffixes 参数。如果`left_df`和`right_df`有共同的列名，`_x`被添加到`left_df`中各自的列名，`_y`被添加到`right_df`中相同的列名。您也可以使用`suffixes`参数手动指定后缀。
*   `indicator`参数用于指示输入数据帧中是否存在连接键。如果`indicator`被设置为`True`，名为`“_merge”`的附加列被添加到输出数据帧。对于每一行，如果连接键同时出现在`left_df`和`right_df`中，则输出为`both`中的`_merge`列。如果连接键仅出现在`left_df`或`right_df`中，则`_merge`列中的值将分别为`left_only`或`right_only`。

执行后，`merge()`函数返回输出数据帧。

## 使用 Merge()函数合并或内部连接数据帧

我们可以使用`merge()`函数合并两个数据帧。合并功能类似于数据库连接操作。连接数据帧时比较的列被传递给`left_on`和`right_on`参数。

在比较左数据帧中的`left_on`列和右数据帧中的`right_on`列的值后，将这些行合并以产生输出数据帧。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,left_on="Roll", right_on="Roll")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
   Roll  Marks Grade  Class      Name
0    11     85     A      1    Aditya
1    12     95     A      1     Chris
2    13     75     B      1       Sam
3    14     75     B      1      Joel
4    16     78     B      1  Samantha
5    15     55     C      1       Tom
6    20     72     B      1      Tina
7    24     92     A      1       Amy
```

在上面的例子中，我们已经使用`"Roll"`属性对熊猫数据帧执行了[内部连接。从两个数据帧中，`"Roll"`属性具有相同值的行被合并在一起，以形成输出数据帧的行。](https://www.pythonforbeginners.com/basics/inner-join-dataframes-in-python)

如果输入数据帧有公共列，那么在连接操作之后，后缀`_x`和`_y`分别被添加到左和右熊猫数据帧的列名中。您可以在下面的示例中观察到这一点。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_name.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,left_on="Roll", right_on="Roll")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
   Roll    Name_x  Marks Grade  Class    Name_y
0    11    Aditya     85     A      1    Aditya
1    12     Chris     95     A      1     Chris
2    13       Sam     75     B      1       Sam
3    14      Joel     75     B      1      Joel
4    16       Tom     78     B      1  Samantha
5    15  Samantha     55     C      1       Tom
6    20      Tina     72     B      1      Tina
7    24       Amy     92     A      1       Amy
```

在这个例子中，两个输入数据帧都有 `"Name`属性。因此，第一个数据帧的`Name`列在输出数据帧中得到`_x`后缀。类似地，第二个数据帧的`Name`列在输出数据帧中得到后缀`_y`。

除了默认后缀，我们还可以更改后缀，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_name.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,left_on="Roll", right_on="Roll",suffixes=("_left","_right"))
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
   Roll Name_left  Marks Grade  Class Name_right
0    11    Aditya     85     A      1     Aditya
1    12     Chris     95     A      1      Chris
2    13       Sam     75     B      1        Sam
3    14      Joel     75     B      1       Joel
4    16       Tom     78     B      1   Samantha
5    15  Samantha     55     C      1        Tom
6    20      Tina     72     B      1       Tina
7    24       Amy     92     A      1        Amy
```

在上面的例子中，我们对第一个数据帧使用了`_left`后缀，对第二个数据帧使用了`_right`后缀。您也可以使用`"suffixes"`参数更改后缀。

### 使用 merge()函数左连接熊猫数据帧

默认情况下，`merge()`函数对输入数据帧执行内部连接操作。我们也可以使用左连接操作来[合并数据帧。为此，我们可以将文字`"left"` 传递给`how`参数。](https://www.pythonforbeginners.com/basics/left-join-dataframes-in-python)

当我们对 pandas 数据帧执行 left join 操作时，左侧数据帧的所有行都显示在输出数据帧中。对于左侧数据帧中没有右侧数据帧中对应行的行，我们在对应于右侧数据帧的列的行中获得`NaN`值。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,how="left",left_on="Roll", right_on="Roll",suffixes=("_left","_right"))
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
   Roll  Marks Grade  Class      Name
0    11     85     A    1.0    Aditya
1    12     95     A    1.0     Chris
2    13     75     B    1.0       Sam
3    14     75     B    1.0      Joel
4    16     78     B    1.0  Samantha
5    15     55     C    1.0       Tom
6    19     75     B    NaN       NaN
7    20     72     B    1.0      Tina
8    24     92     A    1.0       Amy
9    25     95     A    NaN       NaN
```

在上面的例子中，我们在右边的数据帧中没有包含`Roll 19`和`Roll 25`的行。但是，它们存在于左边的数据帧中。因此，在左连接操作之后，我们在输出数据帧中得到具有 Roll 19 和 25 的行。然而，右侧数据帧的列在辊 19 和 25 的相应行中具有`NaN`值。

### 使用 merge()函数连接熊猫数据帧

类似于左连接操作，我们也可以使用`merge()`函数对熊猫数据帧执行[右连接操作。我们可以通过将文字`"right"` 传递给`"how"`参数，使用右连接操作来合并数据帧。](https://www.pythonforbeginners.com/basics/right-join-dataframes-in-python)

当我们对 pandas 数据帧执行右连接操作时，右数据帧的所有行都显示在输出数据帧中。对于右侧数据帧中在左侧数据帧中没有对应行的行，我们在对应于左侧数据帧的列的行中获得`NaN`值。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,how="right",left_on="Roll", right_on="Roll",suffixes=("_left","_right"))
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
    Roll  Marks Grade  Class      Name
0     11   85.0     A      1    Aditya
1     12   95.0     A      1     Chris
2     13   75.0     B      1       Sam
3     14   75.0     B      1      Joel
4     15   55.0     C      1       Tom
5     16   78.0     B      1  Samantha
6     17    NaN   NaN      1     Pablo
7     20   72.0     B      1      Tina
8     24   92.0     A      1       Amy
9     30    NaN   NaN      1    Justin
10    31    NaN   NaN      1      Karl
```

在本例中，左侧数据帧中没有包含第 17、30 和 31 卷的行。但是，它们存在于正确的数据框架中。因此，在右连接操作之后，我们在输出数据帧中得到具有 Roll 17、30 和 31 的行。然而，左侧数据帧的列在辊 17、30 和 31 的相应行中具有`NaN`值。

## 使用 merge()函数完全外部连接 Pandas 数据帧

pandas 数据帧上的内连接、左连接和右连接操作会导致数据丢失。如果希望保留所有输入数据，可以在 pandas 数据帧上执行完全外部连接。为此，您需要将文字`"outer"`传递给`merge()`函数中的`"how"`参数。

在外部连接操作中，左侧数据帧的所有行都显示在输出数据帧中。对于左侧数据帧中没有右侧数据帧中对应行的行，我们在对应于右侧数据帧的列的行中获得`NaN`值。

类似地，右侧数据帧的所有行都显示在输出数据帧中。对于右侧数据帧中在左侧数据帧中没有对应行的行，我们在对应于左侧数据帧的列的行中获得`NaN`值。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,how="outer",left_on="Roll", right_on="Roll",suffixes=("_left","_right"))
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
    Roll  Marks Grade  Class      Name
0     11   85.0     A    1.0    Aditya
1     12   95.0     A    1.0     Chris
2     13   75.0     B    1.0       Sam
3     14   75.0     B    1.0      Joel
4     16   78.0     B    1.0  Samantha
5     15   55.0     C    1.0       Tom
6     19   75.0     B    NaN       NaN
7     20   72.0     B    1.0      Tina
8     24   92.0     A    1.0       Amy
9     25   95.0     A    NaN       NaN
10    17    NaN   NaN    1.0     Pablo
11    30    NaN   NaN    1.0    Justin
12    31    NaN   NaN    1.0      Karl
```

在上面的示例中，我们在右侧数据帧中没有包含第 19 卷和第 25 卷的行。但是，它们存在于左边的数据帧中。因此，在外部连接操作之后，我们在输出数据帧中得到具有 Roll 19 和 25 的行。然而，右侧数据帧的列在辊 19 和 25 的相应行中具有`NaN`值。

类似地，左侧数据帧中没有包含第 17、30 和 31 卷的行。但是，它们存在于正确的数据框架中。因此，在 n 外部连接操作之后，我们在输出数据帧中得到具有 Roll 17、30 和 31 的行。然而，左侧数据帧的列在辊 17、30 和 31 的相应行中具有`NaN`值。

### 使用索引作为连接键合并数据帧

我们也可以使用索引作为连接键来合并数据帧。为此，我们需要根据情况将`left_index`和`right_index`参数设置为 True。

例如，我们可以使用左边数据帧的索引作为连接键，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,left_index=True, right_on="Roll")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
    Class  Roll      Name
0       1    11    Aditya
1       1    12     Chris
2       1    13       Sam
3       1    14      Joel
4       1    15       Tom
5       1    16  Samantha
6       1    17     Pablo
7       1    20      Tina
8       1    24       Amy
9       1    30    Justin
10      1    31      Karl
Merged dataframe is:
   Marks Grade  Class  Roll      Name
0     85     A      1    11    Aditya
1     95     A      1    12     Chris
2     75     B      1    13       Sam
3     75     B      1    14      Joel
5     78     B      1    16  Samantha
4     55     C      1    15       Tom
7     72     B      1    20      Tina
8     92     A      1    24       Amy
```

在本例中，`Roll`列用作左侧数据帧中的索引。因此，我们不需要将列名传递给`left_on`参数。背景`left_index=True`为我们做了工作。

我们也可以使用右边数据帧的索引作为连接键，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,left_on="Roll", right_index=True)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
   Roll  Marks Grade  Class      Name
0    11     85     A      1    Aditya
1    12     95     A      1     Chris
2    13     75     B      1       Sam
3    14     75     B      1      Joel
4    16     78     B      1  Samantha
5    15     55     C      1       Tom
7    20     72     B      1      Tina
8    24     92     A      1       Amy
```

在这个例子中，`Roll`列被用作右边数据帧中的索引。因此，我们不需要将列名传递给`right_on`参数。背景`right_index=True` 为我们做了工作。对于左侧数据帧，我们使用了`left_on`参数来指定连接键。

要使用两个数据帧的索引，我们可以使用如下所示的`left_index`和`right_index`参数。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,left_index=True, right_index=True)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
      Marks Grade  Class      Name
Roll                              
11       85     A      1    Aditya
12       95     A      1     Chris
13       75     B      1       Sam
14       75     B      1      Joel
16       78     B      1  Samantha
15       55     C      1       Tom
20       72     B      1      Tina
24       92     A      1       Amy
```

在这个例子中，两个输入数据帧都有`Roll`列作为它们的索引。因此，我们将`left_index`和`right_index`设置为`True`，以将索引指定为连接键。

### [用指示器合并熊猫数据帧](https://www.pythonforbeginners.com/basics/merge-dataframes-in-python)

在合并数据帧时，我们可以在输出数据帧中添加一些元数据。例如，我们可以指定连接键是否出现在左数据帧、右数据帧或两个输入数据帧中。

为此，我们可以使用指示器参数。当指示器设置为`True`时，`merge()`功能会在输出数据帧中添加一个名为`_merge`的附加列。

对于`_merge`列中的每一行，如果连接键出现在两个输入数据帧中，则输出值出现在`"both"`的`_merge`列中。如果连接键只出现在左侧数据帧中，则`_merge`列的输出为`"left_only"`。类似地，如果联接列只出现在右侧数据帧中，则对应行的`_merge`列中的值将是`"right_only"`。您可以在下面的示例中观察到这一点。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=pd.merge(df1,df2,how="outer",left_index=True, right_index=True,indicator=True)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
      Marks Grade  Class      Name      _merge
Roll                                          
11     85.0     A    1.0    Aditya        both
12     95.0     A    1.0     Chris        both
13     75.0     B    1.0       Sam        both
14     75.0     B    1.0      Joel        both
15     55.0     C    1.0       Tom        both
16     78.0     B    1.0  Samantha        both
17      NaN   NaN    1.0     Pablo  right_only
19     75.0     B    NaN       NaN   left_only
20     72.0     B    1.0      Tina        both
24     92.0     A    1.0       Amy        both
25     95.0     A    NaN       NaN   left_only
30      NaN   NaN    1.0    Justin  right_only
31      NaN   NaN    1.0      Karl  right_only
```

建议阅读:如果你对机器学习感兴趣，可以阅读这篇关于机器学习中[回归的文章。您可能也会喜欢这篇关于带有数字示例](https://codinginfinite.com/regression-in-machine-learning-with-examples/)的 [k 均值聚类的文章。](https://codinginfinite.com/k-means-clustering-using-sklearn-in-python/)

## Pandas concat()函数

我们可以使用 `concat()`函数[水平或垂直连接熊猫数据帧](https://www.pythonforbeginners.com/basics/concatenate-dataframes-in-python)。`concat()`函数的语法如下。

```py
pandas.concat(objs, axis=0, join='outer', ignore_index=False, keys=None,names=None)
```

这里，

*   `objs`参数是需要连接的数据帧的列表或元组。
*   `axis`参数用于决定输入数据帧是水平连接还是垂直连接。如果要垂直连接数据帧，轴设置为默认值 0。要水平连接数据帧，轴设置为 1。
*   `join`参数用于决定如何处理其他索引上的索引。如果我们垂直连接数据帧，join 参数决定输出数据帧中包含哪些列。如果 join 设置为`“inner”`，则输出数据帧只包含那些出现在所有输入数据帧中的列。如果将 join 设置为默认值`“outer”,`，则输入数据帧中的所有列都包含在输出数据帧中。如果我们水平连接数据帧，那么连接操作是用输入数据帧的索引来执行的。
*   `ignore_index`参数用于决定输出数据帧是否存储输入数据帧的索引。默认情况下，它被设置为`False`，因此输入数据帧的索引被保存在输出数据帧中。当`ignore_index`设置为`True`时，输入数据帧的索引被忽略。
*   `keys`参数用于在输出数据帧中创建额外的索引级别。通常，我们使用该参数来标识输出数据帧中的行或列所属的输入数据帧。这是通过将输入数据帧的名称指定为键来实现的。
*   `names`参数用于命名使用`keys`参数创建的索引列。

执行后，`concat()`函数返回连接的数据帧。

### 使用 concat()函数垂直连接数据帧

如果我们有两个具有相似数据的数据帧，我们可以使用`concat()`函数垂直连接它们。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade2.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2])
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Class  Roll    Name  Marks Grade
0      1    11  Aditya     85     A
1      1    12   Chris     95     A
2      1    14     Sam     75     B
3      1    16  Aditya     78     B
4      1    15   Harry     55     C
5      2     1    Joel     68     B
6      2    22     Tom     73     B
7      2    15    Golu     79     B
second dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
Merged dataframe is:
   Class  Roll        Name  Marks Grade
0      1    11      Aditya     85     A
1      1    12       Chris     95     A
2      1    14         Sam     75     B
3      1    16      Aditya     78     B
4      1    15       Harry     55     C
5      2     1        Joel     68     B
6      2    22         Tom     73     B
7      2    15        Golu     79     B
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
```

在上面的示例中，我们在合并所有行后，从输入数据帧创建了输出数据帧。您可以观察到输入数据帧的索引也被连接到输出数据帧中。

要忽略输入数据帧的索引并创建新的索引，可以将`concat()`函数中的`ignore_index`参数设置为`True`，如下例所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade2.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],ignore_index=True)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Class  Roll    Name  Marks Grade
0      1    11  Aditya     85     A
1      1    12   Chris     95     A
2      1    14     Sam     75     B
3      1    16  Aditya     78     B
4      1    15   Harry     55     C
5      2     1    Joel     68     B
6      2    22     Tom     73     B
7      2    15    Golu     79     B
second dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
Merged dataframe is:
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
```

在这里，您可以看到所有的行都被赋予了新的索引。

有时，水平连接的数据帧可能没有相同的列。在这种情况下，输出数据帧中的列是输入数据帧中所有列的并集。例如，考虑下面的例子。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2])
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    20     72     B
7    24     92     A
second dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
Merged dataframe is:
   Roll  Marks Grade      Name
0    11     85     A       NaN
1    12     95     A       NaN
2    13     75     B       NaN
3    14     75     B       NaN
4    16     78     B       NaN
5    15     55     C       NaN
6    20     72     B       NaN
7    24     92     A       NaN
0    11     85     A    Aditya
1    12     95     A     Chris
2    13     75     B       Sam
3    14     75     B      Joel
4    16     78     B       Tom
5    15     55     C  Samantha
6    20     72     B      Tina
7    24     92     A       Amy
```

在上面的例子中，第一个数据帧包含列`"Roll", "Marks",` 和 `"Grade"`。第二个数据帧包含列`"Roll", "Name", "Marks",` 和 `"Grade"`。因此，输出数据帧包含列`"Roll", "Name", "Marks",` 和 `"Grade"`。

对应于第一个数据帧的行中的“`Name`”列中的值将包含`NaN`值，因为第一个数据帧不包含`"Name"`列。类似地，如果第二个数据帧具有额外的列，则对应于第一个数据帧的行将包含相应列的`NaN`值。

### 使用 Concat()函数水平连接数据帧

我们也可以使用`concat()`函数水平连接数据帧。为此，我们需要使用`concat()`函数中的参数`axis=1` 。

当使用`concat()`功能水平连接两个数据帧时，使用数据帧的索引值合并输入数据帧的行。在执行了`concat()`函数之后，我们得到了如下所示的输出数据帧。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],axis=1)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    20     72     B
7    24     92     A
second dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
Merged dataframe is:
   Roll  Marks Grade  Roll      Name  Marks Grade
0    11     85     A    11    Aditya     85     A
1    12     95     A    12     Chris     95     A
2    13     75     B    13       Sam     75     B
3    14     75     B    14      Joel     75     B
4    16     78     B    16       Tom     78     B
5    15     55     C    15  Samantha     55     C
6    20     72     B    20      Tina     72     B
7    24     92     A    24       Amy     92     A
```

如果任何输入数据帧包含其它数据帧中不存在的额外的行或索引，则输出数据帧在对应于相应行中的其它数据帧的列中包含`NaN`值。您可以在下面的示例中观察到这一点。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],axis=1)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    20     72     B
7    24     92     A
second dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
8    25      Sinu     95     A
Merged dataframe is:
   Roll  Marks Grade  Roll      Name  Marks Grade
0  11.0   85.0     A    11    Aditya     85     A
1  12.0   95.0     A    12     Chris     95     A
2  13.0   75.0     B    13       Sam     75     B
3  14.0   75.0     B    14      Joel     75     B
4  16.0   78.0     B    16       Tom     78     B
5  15.0   55.0     C    15  Samantha     55     C
6  20.0   72.0     B    20      Tina     72     B
7  24.0   92.0     A    24       Amy     92     A
8   NaN    NaN   NaN    25      Sinu     95     A
```

在本例中，您可以看到最后一行包含与第一个数据帧对应的列的`NaN`值。这是因为第一个数据帧比第二个数据帧少包含一行。

### 仅用公共行索引水平连接数据帧

如果不希望`NaN`值出现在输出数据帧中，可以使用如下所示的`join="inner"` 参数。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],join="inner",axis=1)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    20     72     B
7    24     92     A
second dataframe is:
   Roll      Name  Marks Grade
0    11    Aditya     85     A
1    12     Chris     95     A
2    13       Sam     75     B
3    14      Joel     75     B
4    16       Tom     78     B
5    15  Samantha     55     C
6    20      Tina     72     B
7    24       Amy     92     A
8    25      Sinu     95     A
Merged dataframe is:
   Roll  Marks Grade  Roll      Name  Marks Grade
0    11     85     A    11    Aditya     85     A
1    12     95     A    12     Chris     95     A
2    13     75     B    13       Sam     75     B
3    14     75     B    14      Joel     75     B
4    16     78     B    16       Tom     78     B
5    15     55     C    15  Samantha     55     C
6    20     72     B    20      Tina     72     B
7    24     92     A    24       Amy     92     A
```

### 用自定义索引水平连接数据帧

当我们水平连接数据帧时，使用默认索引来匹配行。如果我们指定输入数据帧的索引，则使用索引值作为关键字来匹配数据帧的行。您可以在下面的示例中观察到这一点。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],join="inner",axis=1)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
20       72     B
24       92     A
second dataframe is:
          Name  Marks Grade
Roll                       
11      Aditya     85     A
12       Chris     95     A
13         Sam     75     B
14        Joel     75     B
16         Tom     78     B
15    Samantha     55     C
20        Tina     72     B
24         Amy     92     A
25        Sinu     95     A
Merged dataframe is:
      Marks Grade      Name  Marks Grade
Roll                                    
11       85     A    Aditya     85     A
12       95     A     Chris     95     A
13       75     B       Sam     75     B
14       75     B      Joel     75     B
16       78     B       Tom     78     B
15       55     C  Samantha     55     C
20       72     B      Tina     72     B
24       92     A       Amy     92     A
```

### 在连接数据帧后向行添加标识符

当我们连接两个数据帧时，输出数据帧并不指定特定行属于哪个输入数据帧。要指定这一点，我们可以使用`"keys"`参数。`"keys"`参数接受一个字符串列表作为它的输入。执行后，它将关键字作为额外级别的索引添加到输出数据帧中。在`"keys"`参数中指定的每个键对应一个特定的数据帧。您可以在下面的示例中观察到这一点。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade2.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],keys=["Dataframe1","dataframe2"])
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Class  Roll    Name  Marks Grade
0      1    11  Aditya     85     A
1      1    12   Chris     95     A
2      1    14     Sam     75     B
3      1    16  Aditya     78     B
4      1    15   Harry     55     C
5      2     1    Joel     68     B
6      2    22     Tom     73     B
7      2    15    Golu     79     B
second dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
Merged dataframe is:
              Class  Roll        Name  Marks Grade
Dataframe1 0      1    11      Aditya     85     A
           1      1    12       Chris     95     A
           2      1    14         Sam     75     B
           3      1    16      Aditya     78     B
           4      1    15       Harry     55     C
           5      2     1        Joel     68     B
           6      2    22         Tom     73     B
           7      2    15        Golu     79     B
dataframe2 0      2    27       Harsh     55     C
           1      2    23       Clara     78     B
           2      3    33        Tina     82     A
           3      3    34         Amy     88     A
           4      3    15    Prashant     78     B
           5      3    27      Aditya     55     C
           6      3    23  Radheshyam     78     B
           7      3    11       Bobby     50     D
```

在上面的示例中，您可以看到我们在输出数据帧中添加了“`dataframe1`”和“`dataframe2`”作为额外的索引级别。您还可以使用如下所示的“`names`”参数为索引列命名。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade2.csv")
print("second dataframe is:")
print(df2)
df3=pd.concat([df1,df2],keys=["Dataframe1","dataframe2"],names=["Dataframe","index"])
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Class  Roll    Name  Marks Grade
0      1    11  Aditya     85     A
1      1    12   Chris     95     A
2      1    14     Sam     75     B
3      1    16  Aditya     78     B
4      1    15   Harry     55     C
5      2     1    Joel     68     B
6      2    22     Tom     73     B
7      2    15    Golu     79     B
second dataframe is:
   Class  Roll        Name  Marks Grade
0      2    27       Harsh     55     C
1      2    23       Clara     78     B
2      3    33        Tina     82     A
3      3    34         Amy     88     A
4      3    15    Prashant     78     B
5      3    27      Aditya     55     C
6      3    23  Radheshyam     78     B
7      3    11       Bobby     50     D
Merged dataframe is:
                  Class  Roll        Name  Marks Grade
Dataframe  index                                      
Dataframe1 0          1    11      Aditya     85     A
           1          1    12       Chris     95     A
           2          1    14         Sam     75     B
           3          1    16      Aditya     78     B
           4          1    15       Harry     55     C
           5          2     1        Joel     68     B
           6          2    22         Tom     73     B
           7          2    15        Golu     79     B
dataframe2 0          2    27       Harsh     55     C
           1          2    23       Clara     78     B
           2          3    33        Tina     82     A
           3          3    34         Amy     88     A
           4          3    15    Prashant     78     B
           5          3    27      Aditya     55     C
           6          3    23  Radheshyam     78     B
           7          3    11       Bobby     50     D
```

## 熊猫加入()方法

代替`merge()`函数，我们也可以使用`join()`方法在两个数据帧上执行连接。

`join()`方法的语法如下。

```py
df.join(other, on=None, how='left', lsuffix='', rsuffix='’)
```

这里，

*   是我们左边的数据帧。
*   如果我们想要连接多个数据帧，参数`“other”`表示要连接的正确数据帧或数据帧列表。
*   `on`参数用于指定用作连接键的列。
*   “`how`”参数用于指定是执行左连接、右连接、内连接还是全外连接。默认情况下，它的值为“`left`”。
*   当左右数据帧具有共同的列名时，`lsuffix`和`rsuffix`参数用于指定输出数据帧中来自输入数据帧的列的后缀。

当使用数据帧的索引作为连接键执行连接时,`join()` 方法工作得最好。

## 使用 Join()方法左连接熊猫数据帧

默认情况下， `join()`方法执行左连接操作。当在一个数据帧上调用时，它将另一个数据帧作为其输入参数。执行后，它返回连接的数据帧。使用索引列作为连接键来连接输入数据帧。

在这里，我想强调的是，`join()`方法最适合具有定制索引列的数据帧。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=df1.join(df2)
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
      Marks Grade  Class      Name
Roll                              
11       85     A    1.0    Aditya
12       95     A    1.0     Chris
13       75     B    1.0       Sam
14       75     B    1.0      Joel
16       78     B    1.0  Samantha
15       55     C    1.0       Tom
19       75     B    NaN       NaN
20       72     B    1.0      Tina
24       92     A    1.0       Amy
25       95     A    NaN       NaN
```

## 使用 Join()方法右连接熊猫数据帧

我们可以使用如下所示的`join()`方法对 pandas 数据帧执行右连接操作。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=df1.join(df2,how="right")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
      Marks Grade  Class      Name
Roll                              
11     85.0     A      1    Aditya
12     95.0     A      1     Chris
13     75.0     B      1       Sam
14     75.0     B      1      Joel
15     55.0     C      1       Tom
16     78.0     B      1  Samantha
17      NaN   NaN      1     Pablo
20     72.0     B      1      Tina
24     92.0     A      1       Amy
30      NaN   NaN      1    Justin
31      NaN   NaN      1      Karl
```

## 使用 Join()方法内部连接数据帧

我们可以通过将参数`how="inner"`传递给 `join()` 方法来对 pandas 数据帧执行内部连接操作，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=df1.join(df2,how="inner")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
      Marks Grade  Class      Name
Roll                              
11       85     A      1    Aditya
12       95     A      1     Chris
13       75     B      1       Sam
14       75     B      1      Joel
16       78     B      1  Samantha
15       55     C      1       Tom
20       72     B      1      Tina
24       92     A      1       Amy
```

## 使用 Join()方法的外部连接数据帧

我们可以使用如下例所示的`join()`方法在 pandas 数据帧上执行外部连接。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=df1.join(df2,how="outer")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
      Marks Grade  Class      Name
Roll                              
11     85.0     A    1.0    Aditya
12     95.0     A    1.0     Chris
13     75.0     B    1.0       Sam
14     75.0     B    1.0      Joel
15     55.0     C    1.0       Tom
16     78.0     B    1.0  Samantha
17      NaN   NaN    1.0     Pablo
19     75.0     B    NaN       NaN
20     72.0     B    1.0      Tina
24     92.0     A    1.0       Amy
25     95.0     A    NaN       NaN
30      NaN   NaN    1.0    Justin
31      NaN   NaN    1.0      Karl
```

## 用公共列名连接数据框架

如果输入数据帧有共同的列名，我们需要使用参数`rsuffix`和`lsuffix`为数据帧的列指定后缀，如下所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=df1.join(df2,on="Roll",how="outer",lsuffix="_left", rsuffix="_right")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
          Name  Marks Grade
Roll                       
11      Aditya     85     A
12       Chris     95     A
13         Sam     75     B
14        Joel     75     B
16         Tom     78     B
15    Samantha     55     C
20        Tina     72     B
24         Amy     92     A
25        Sinu     95     A
Merged dataframe is:
   Roll  Marks_left Grade_left      Name  Marks_right Grade_right
0    11          85          A    Aditya         85.0           A
1    12          95          A     Chris         95.0           A
2    13          75          B       Sam         75.0           B
3    14          75          B      Joel         75.0           B
4    16          78          B       Tom         78.0           B
5    15          55          C  Samantha         55.0           C
6    19          75          B       NaN          NaN         NaN
7    20          72          B      Tina         72.0           B
8    24          92          A       Amy         92.0           A
9    25          95          A      Sinu         95.0           A
```

在这个例子中，`Marks`和`Grade`列出现在两个输入数据帧中。因此，我们需要为列名指定后缀。

如果不指定后缀，并且输入数据帧有公共的列名，程序就会出错。

## 使用列名作为连接键连接数据帧

如果调用了`join()`方法的 dataframe 没有连接键作为索引列，您可以使用`"on"`参数指定列名。但是，作为输入参数传递的 dataframe 需要将用作连接键的列作为其索引。您可以在下面的示例中观察到这一点。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data1.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=df1.join(df2,on="Roll",how="outer")
print("Merged dataframe is:")
print(df3)
```

输出:

```py
First dataframe is:
   Roll  Marks Grade
0    11     85     A
1    12     95     A
2    13     75     B
3    14     75     B
4    16     78     B
5    15     55     C
6    19     75     B
7    20     72     B
8    24     92     A
9    25     95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
17        1     Pablo
20        1      Tina
24        1       Amy
30        1    Justin
31        1      Karl
Merged dataframe is:
     Roll  Marks Grade  Class      Name
0.0    11   85.0     A    1.0    Aditya
1.0    12   95.0     A    1.0     Chris
2.0    13   75.0     B    1.0       Sam
3.0    14   75.0     B    1.0      Joel
4.0    16   78.0     B    1.0  Samantha
5.0    15   55.0     C    1.0       Tom
6.0    19   75.0     B    NaN       NaN
7.0    20   72.0     B    1.0      Tina
8.0    24   92.0     A    1.0       Amy
9.0    25   95.0     A    NaN       NaN
NaN    17    NaN   NaN    1.0     Pablo
NaN    30    NaN   NaN    1.0    Justin
NaN    31    NaN   NaN    1.0      Karl
```

## 使用 Join()方法连接多个数据帧

我们也可以使用`join()` 方法连接多个数据帧。为此，我们需要在一个数据帧上调用`join()`方法，并将列表中的其他数据帧作为输入传递给`join()`方法。

这里，我们需要确保所有的数据帧都应该将连接键作为它们的索引列。此外，除了索引列之外，任何两个数据帧之间都不应该有一个公共列。遵循这两个条件，我们可以使用如下所示的`join()` 方法连接多个 pandas 数据帧。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("name_data.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=pd.read_csv("height_with_roll.csv",index_col="Roll")
print("Third dataframe is:")
print(df3)
df4=df1.join([df2,df3])
print("Merged dataframe is:")
print(df4)
```

输出:

```py
First dataframe is:
      Marks Grade
Roll             
11       85     A
12       95     A
13       75     B
14       75     B
16       78     B
15       55     C
19       75     B
20       72     B
24       92     A
25       95     A
second dataframe is:
      Class      Name
Roll                 
11        1    Aditya
12        1     Chris
13        1       Sam
14        1      Joel
15        1       Tom
16        1  Samantha
20        1      Tina
24        1       Amy
Third dataframe is:
      Height
Roll        
11       170
12       165
13       155
14       180
16       140
15       162
19       175
20       163
24       154
25       161
Merged dataframe is:
      Marks Grade  Class      Name  Height
Roll                                      
11       85     A    1.0    Aditya     170
12       95     A    1.0     Chris     165
13       75     B    1.0       Sam     155
14       75     B    1.0      Joel     180
16       78     B    1.0  Samantha     140
15       55     C    1.0       Tom     162
19       75     B    NaN       NaN     175
20       72     B    1.0      Tina     163
24       92     A    1.0       Amy     154
25       95     A    NaN       NaN     161
```

如果输入数据帧有共同的列名，程序将运行到`ValueError`异常，如下例所示。

```py
import numpy as np
import pandas as pd
df1=pd.read_csv("grade_with_roll1.csv",index_col="Roll")
print("First dataframe is:")
print(df1)
df2=pd.read_csv("grade_with_name.csv",index_col="Roll")
print("second dataframe is:")
print(df2)
df3=pd.read_csv("height_with_roll.csv",index_col="Roll")
print("Third dataframe is:")
print(df3)
df4=df1.join([df2,df3])
print("Merged dataframe is:")
print(df4)
```

输出:

```py
ValueError: Indexes have overlapping values: Index(['Marks', 'Grade'], dtype='object')
```

## 结论

在本文中，我们讨论了如何在 python 中合并、连接和联接 pandas 数据帧。。要了解更多关于 python 编程的知识，你可以阅读这篇关于如何[创建熊猫数据框架](https://www.pythonforbeginners.com/basics/create-pandas-dataframe-in-python)的文章。您可能也会喜欢这篇关于 Python 中的[字符串操作的文章。](https://www.pythonforbeginners.com/basics/string-manipulation-in-python)