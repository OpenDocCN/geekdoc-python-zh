# Python 中的布尔索引——快速指南

> 原文：<https://www.askpython.com/python/examples/boolean-indexing>

布尔值可以作为数据帧中的索引，这不是很有趣吗？布尔索引代表数据帧中的每一行。布尔索引可以帮助我们从数据集中过滤掉不必要的数据。过滤数据可以让您获得一些深入的信息，否则您可能无法找到这些信息。在本文中，我们将学习如何使用布尔索引来过滤和分割数据。所以让我们开始吧！

## Python 中的布尔索引

让我们从创建一个数据帧开始。我们将使用参加竞赛的一组候选人的年龄数据创建一个数据框架。

```py
import pandas as pd
# Creating a dictionary
data = {'Name':["Tommy","Linda","Justin","Brendon"], 'Age':[31,24,16,22]}
df = pd.DataFrame(data,index=[True,False,True,False])
print(df)

```

**输出**

```py
        Name         Age
True     Tommy   31
False    Linda   24
True    Justin   16
False  Brendon   22

```

### 1.使用. loc []函数

这是一个优秀而简单的函数，可以帮助你根据布尔索引过滤数据。使用这个函数，我们可以过滤掉具有特定布尔值的数据。假设我们将 True 传递给。loc []函数，我们将只得到索引值为 True 的过滤数据。在这个方法中，我们不能使用整数作为布尔值。

**例如:**

```py
import pandas as pd
# Creating a dictionary
data = {'Name':["Tommy","Linda","Justin","Brendon"], 'Age':[31,24,16,22]}
df = pd.DataFrame(data,index=[True,False,True,False])
print(df.loc[True])

```

**输出**:

```py
        Name       Age
True   Tommy   31
True  Justin   16

```

### 2.使用。iloc[]函数

iloc[]函数只接受整数值，因此我们需要将整数值传递给该函数。

**例如:**

```py
import pandas as pd
# Creating a dictionary
data = {'Name':["Tommy","Linda","Justin","Brendon"], 'Age':[31,24,16,22]}
df = pd.DataFrame(data,index=[1,0,0,1])
print(df.iloc[1])

```

**输出:**

```py
Name    Linda
Age        24
Name: 0, dtype: object

```

### 3.使用。ix[]函数

这也是一种类似于上面的方法，但是在这种情况下我们可以使用整数作为布尔值。因此，例如，如果我们将索引值指定为 1 和 0，我们可以过滤索引值为 0 或 1 的行。

```py
import pandas as pd
# Creating a dictionary
data = {'Name':["Tommy","Linda","Justin","Brendon"], 'Age':[31,24,16,22]}
df = pd.DataFrame(data,index=[1,1,0,0])
print(df.ix[0])

```

**输出:**

```py
           Name       Age
0       Justin          16
0       Brendon     22

```

## 结论

总之，我们学习了如何在 python 中使用布尔索引并过滤有用的数据。希望这篇文章对你有所帮助。