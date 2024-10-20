# Python loc()函数–从数据集中提取值

> 原文：<https://www.askpython.com/python-modules/pandas/python-loc-function>

嘿读者们！在本文中，我们将详细关注 **Python loc()函数**的功能。所以，让我们开始吧！！

* * *

## Python loc()函数的工作原理

Python 由各种模块组成，这些模块具有处理和操作数据值的内置函数。

一个这样的模块是熊猫模块。

Pandas 模块使我们能够处理包含大量数据的大型数据集。

这就是`Python loc() function`出现的时候。loc()函数帮助我们轻松地从数据集中检索数据值。

使用 loc()函数，我们可以根据传递给该函数的索引值来访问适合特定行或列的数据值。

**语法:**

```py
pandas.DataFrame.loc[index label]

```

我们需要提供索引值，以便在输出中表示整个数据。

**索引标签可能是下列值之一**:

*   单一标签–示例:字符串
*   字符串列表
*   带标签的切片对象
*   标签的[数组](https://www.askpython.com/python/array/python-array-declaration)的列表等。

因此，我们可以使用 loc()函数基于索引标签从数据集中检索特定的记录。

注意:如果传递的索引没有作为标签出现，它返回 **KeyError** 。

现在让我们使用下面的例子来关注相同的实现。

* * *

## Python loc()函数示例

让我们首先使用 Pandas 模块中的数据框创建一个包含一组数据值的数据框，如下所示:

```py
import pandas as pd
data = pd.DataFrame([[1,1,1], [4,4,4], [7,7,7], [10,10,10]],
     index=['Python', 'Java', 'C','Kotlin'],
     columns=['RATE','EE','AA'])
print(data)

```

**数据帧**:

```py
	RATE	EE	AA
Python	1	1	1
Java	4	4	4
C	7	7	7
Kotlin	10	10	10

```

创建了具有一组定义值的数据框后，现在让我们尝试检索一组包含特定索引的数据值的行或列，如下所示:

### **从数据帧中提取一行**

```py
print(data.loc['Python'])

```

因此，使用上面的命令，我们已经提取了与索引标签‘Python’相关联的所有数据值。

**输出:**

```py
RATE    1
EE      1
AA      1
Name: Python, dtype: int64

```

### **从一个数据帧中提取多行**

现在让我们尝试使用下面的命令同时提取与多个索引相关联的数据行和数据列。

```py
print(data.loc[['Python','C']])

```

**输出:**

```py
          RATE  EE  AA
Python     1    1    1
C          7    7    7

```

### 使用 Python loc()提取一系列行

```py
print(data.loc['Python':'C'])

```

这里，我们将 slice 对象与标签一起使用，以显示与从‘Python’到‘C’的标签相关联的行和列。

**输出:**

```py
          RATE  EE  AA
Python     1   1   1
Java       4   4   4
C          7   7   7

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，欢迎在下面评论。

更多与 Python 相关的帖子，敬请关注，祝你学习愉快！！

* * *

## 参考

*   [Python pandas.loc()函数—文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)