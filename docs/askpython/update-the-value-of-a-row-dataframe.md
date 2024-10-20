# 如何在 Python 数据帧中更新行的值？

> 原文：<https://www.askpython.com/python-modules/pandas/update-the-value-of-a-row-dataframe>

读者朋友们，你们好！在这篇文章中，我们将详细讨论用不同的方法来更新 Python 数据帧中的一行的值。

所以，让我们开始吧！

* * *

## 首先，行和列位于哪里？

在 Python 编程语言中，我们遇到了这个叫做 [Pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 的模块，它为我们提供了一个叫做数据帧的数据结构。

数据框以行和列的形式存储数据。因此，它可以被视为一个矩阵，在分析数据时非常有用。

让我们马上创建一个数据框架！

```py
import pandas as pd 
info= {"Num":[12,14,13,12,14,13,15], "NAME":['John','Camili','Rheana','Joseph','Amanti','Alexa','Siri']}

data = pd.DataFrame(info)
print("Original Data frame:\n")
print(data)

```

这里，我们使用`pandas.DataFrame()`函数创建了一个数据框

**输出:**

```py
Original Data frame:

   Num   NAME
0   12    John
1   14  Camili
2   13  Rheana
3   12  Joseph
4   14  Amanti
5   13   Alexa
6   15    Siri

```

我们将在整篇文章中使用上面创建的数据框作为示例参考。

* * *

## 1.使用 Python at()方法更新行的值

**Python at()方法**使我们能够相对于一列一次更新一行的值。

**语法:**

```py
dataframe.at[index,'column-name']='new value'

```

**举例:**

在本例中，我们为 at()函数提供了数据帧和列“NAME”的索引 6。因此，行索引 6 处的列“NAME”的值得到更新。

```py
data.at[6,'NAME']='Safa'

```

**输出:**

```py
Num    NAME
0   12    John
1   14  Camili
2   13  Rheana
3   12  Joseph
4   14  Amanti
5   13   Alexa
6   15    Safa

```

* * *

## 2.Python loc()函数用于更改行/列的值

[Python loc()方法](https://www.askpython.com/python-modules/pandas/python-loc-function)也可以通过提供列的标签和行的[索引](https://www.askpython.com/python/list/indexing-in-python)来更新与列相关的行的值。

**语法:**

```py
dataframe.loc[row index,['column-names']] = value

```

**举例:**

```py
data.loc[0:2,['Num','NAME']] = [100,'Python']

```

这里，我们已经分别针对列“Num”和“NAME”更新了从索引 0 到 2 的行的值。

**输出:**

```py
Num    NAME
0  100  Python
1  100  Python
2  100  Python
3   12  Joseph
4   14  Amanti
5   13   Alexa
6   15    Siri

```

* * *

## 3.Python replace()方法更新数据帧中的值

使用 [Python replace()方法](https://www.askpython.com/python/string/python-replace-function)，我们可以更新或更改数据框中任何字符串的值。我们不需要向它提供索引或标签值。

**语法:**

```py
dataframe.replace("old string", "new string")

```

**举例:**

```py
data.replace("Siri", 
           "Code", 
           inplace=True)

```

如上所述，我们在数据框中用“代码”替换了“Siri”一词。

**输出:**

```py
 Num    NAME
0   12    John
1   14  Camili
2   13  Rheana
3   12  Joseph
4   14  Amanti
5   13   Alexa
6   15    Code

```

* * *

## 4.使用 iloc()方法更新行的值

使用 [Python iloc()方法](https://www.askpython.com/python/built-in-methods/python-iloc-function)，可以通过提供行/列的索引值来更改或更新行/列的值。

**语法:**

```py
dataframe.iloc[index] = value

```

**举例:**

```py
data.iloc[[0,1,3,6],[0]] = 100

```

在本例中，我们已经更新了第 0、1、3 和 6 行相对于第一列的值，即“Num”为 100。

我们甚至可以使用 iloc()函数为函数提供行切片，从而更改多行的值。

**输出:**

```py
Num    NAME
0  100    John
1  100  Camili
2   13  Rheana
3  100  Joseph
4   14  Amanti
5   13   Alexa
6  100    Siri

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注，在此之前，祝你学习愉快！！🙂