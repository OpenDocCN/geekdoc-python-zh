# Python 熊猫系列:快速指南

> 原文：<https://www.askpython.com/python-modules/pandas/python-pandas-series>

如果你像我一样，那么你喜欢学习新事物。Python 是一种很好的语言，可以做到这一点。Pandas 是用 Python 处理数据的一个很好的库。

在这个快速指南中，我将向您展示如何在 Python 中使用熊猫系列。我们将介绍什么是系列，如何创建系列，以及如何操作和访问系列中的数据。

所以让我们开始吧！

### 什么是系列？

Pandas 系列是存储各种数据类型的数据的一维数组。我们可以像访问数组数据一样，通过索引来访问一个系列的内容。熊猫系列最重要的两个特点如下。

*   一个系列不能包含多列。所以它是一维的。
*   我们可以使用`series()`方法轻松地将列表、元组和字典转换成序列。

可以使用下面的命令创建熊猫系列

```py
pandas.Series( data, index, dtype )

```

上述 python 命令中的参数是:

*   **数据**–它可以是各种形式，如数组、列表或常量。
*   **索引**–该值是唯一的，可散列的，与数据长度相同。默认情况下，如果没有索引通过，根据各自的内容，它**从 0 开始，以此类推。**
*   **dtype**–它是我们输入内容的数据类型。如果没有通过，那就是推断。

## 让我们创造我们的熊猫系列

现在，我们将以不同的方式创建我们的系列，下面是一些例子。

### 从 ndarray 创建系列

#### 示例 1

```py
#We need to import both the pandas and numpy libraries to create a series
import pandas as pd
import numpy as np

data = np.array(['h','e','l','l','o'])
our_series = pd.Series(data)
#our series created named our_series

```

我们可以在上面的代码片段中看到，我们刚刚创建了第一个名为 our_series 的 python 系列。它将创建我们的系列，指数从 0 开始，依此类推。让我们把这个打印出来，看看它是怎么被创造出来的。

```py
print (our_series)
0   h
1   e
2   l
3   l
4   o
dtype: object

```

#### 示例 2

让我们使用相同的方法创建另一个，但是在这个例子中，我们将传递一些手动索引。让我们进入我们的代码片段。

```py
#We need to import both the pandas and numpy libraries to create a series
import pandas as pd
import numpy as np

data = np.array(['h','e','l','l','o'])
our_series = pd.Series(data, index=[100,101,102,103,104])
#our series created named our_series

```

让我们通过打印创建的系列来查看代码片段的输出。

```py
print (our_series)
100   h
101   e
102   l
103   l
104   o
dtype: object

```

这里我们可以看到索引与我们在创建系列时传递的值相同。

### 从字典创建系列

#### 示例 1

```py
#importing the pandas and numpy library 
import pandas as pd
import numpy as np

data = {'a' : 0., 'b' : 1., 'c' : 2.}
our_series = pd.Series(data)
print (our_series)

```

一个字典可以作为输入传递，如果没有指定索引，那么字典键将按排序顺序构造索引。如果索引通过，作为索引的相应数据将优先如下。

```py
print(our_series)
a 0.0
b 1.0
c 2.0
dtype: float64

```

#### 示例 2

```py
#importing the pandas and numpy library 
import pandas as pd
import numpy as np

data = {'a' : 0., 'b' : 1., 'c' : 2.}
our_series = pd.Series(data, index=['b','c','a'])

```

这里，我们传递了与字典的键相对应的手动索引。通过打印创建的系列，我们可以得到如下输出。

```py
print (our_series)
b 1.0
c 2.0
a 0.0
dtype: float64

```

### 从标量创建序列

我们将输入标量值的数据，必须提供一个索引。该值将被重复以匹配索引的长度。我们要用`pandas.Series()`。让我们按照下面的代码片段。

```py
#import the pandas library and numpy library
import pandas as pd
import numpy as np

ssss = pd.Series(5, index=[0, 1, 2, 3])

```

我们可以通过打印新创建的系列来查看结果系列。

```py
print(ssss)
0  5
1  5
2  5
3  5
dtype: int64

```

### 创建空系列

```py
#importing only pandas library as pd
import pandas as pd

s = pd.Series()

```

如果我们打印上面的序列，我们可以得到如下的空序列。

```py
print (s)
Series([], dtype: float64)

```

### 我们将学习如何使用各自的索引来访问系列数据

在这里，我们将检索我们系列的第一个元素。我们已经知道，索引计数从零开始，这意味着第一个元素存储在第零个^(到第)个位置，以此类推。

```py
import pandas as pd
sss = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])
#our series created

#retrieving the first element using his index
print (s['a'])

```

我们将得到如下输出。

```py
1

```

这样，如果我们打印任何其他指数，我们可以得到如下。

```py
print (s[['a','c','d']])
a  1
c  3
d  4
dtype: int64

print (sss[0])
1

print (sss[4])
5

#retieving first three elements
print (sss[:3])
a  1
b  2
c  3
dtype: int64

#retrieving the last three elements
print(sss[-3:])
c  3
d  4
e  5
dtype: int64

```

## 结论

在这个快速指南中，我们了解了 Python 熊猫系列。我们已经了解了如何创建一个系列，如何操作和访问系列中的数据，以及如何对系列执行一些基本操作。我希望这个指南对你有所帮助。如果你有任何问题，请在下面的评论中发表。