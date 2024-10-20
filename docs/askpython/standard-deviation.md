# Python 中标准偏差的 3 种变体

> 原文：<https://www.askpython.com/python/examples/standard-deviation>

嘿，读者们！在本文中，我们将关注 Python 中标准差的 **3 种变体。**

所以在开始之前，让我们先了解一下什么是标准差？

标准偏差表示数据值或实体相对于平均值或中心值的偏差。它主要用于数据分析领域，探索和分析数据分布。

现在，让我们在下一节中进一步看看 Python 中计算标准差的各种方法。

* * *

## 变体 1:使用 stdev()函数的 Python 中的标准偏差

`Python statistics module`为我们提供了`statistics.stdev() function`来计算一组值的标准差。

**语法:**

```py
statistics.stdev(data)

```

在下面的示例中，我们创建了一个列表，并对数据值执行了标准差操作，如下所示

**举例:**

```py
import statistics as std
lst = [1,2,3,4,5]

stat = std.stdev(lst)
print(stat)

```

**输出:**

```py
1.5811388300841898

```

* * *

## 变体 2:使用 NumPy 模块的标准偏差

NumPy 模块为我们提供了各种函数来处理和操作数字数据值。

我们可以使用如下所示的`numpy.std() function`计算数值范围的标准偏差

**语法:**

```py
numpy.std(data)

```

**举例:**

```py
import numpy as np
num = np.arange(1,6)
stat = np.std(num)
print(stat)

```

这里，我们已经利用`numpy.arange() function`生成了一组 1-6 之间的连续值。此外，已经使用 std()函数计算了标准偏差。

**输出:**

```py
1.4142135623730951

```

* * *

## 变量 3:熊猫模块的标准偏差

Pandas 模块使我们能够处理大量的数据集，并为我们提供了在这些数据集上执行的各种功能。

使用 Pandas 模块，我们可以对数据值执行各种统计操作，其中一项是标准差，如下所示

**语法:**

```py
dataframe.std()

```

**举例:**

```py
import pandas as pd
lst = [1,2,3,4,5,6,7]
data = pd.DataFrame(lst)
stat = data.std()
print(stat)

```

在本例中，我们创建了一个[列表](https://www.askpython.com/python/list/python-list)，然后使用 pandas.dataframe()函数将该列表转换为数据框。此外，我们已经使用`std()`函数计算了数据框中出现的那些值的标准偏差。

**输出:**

```py
0    2.160247
dtype: float64

```

* * *

## 结论

到此，我们就结束了这个话题。如果你遇到任何问题，请随时在下面评论。

更多此类与 Python 相关的帖子，敬请关注@ [AskPython](https://www.askpython.com/) 继续学习！