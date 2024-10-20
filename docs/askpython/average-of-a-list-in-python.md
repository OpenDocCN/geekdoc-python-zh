# 用 5 种简单的方法在 Python 中求一个列表的平均值

> 原文：<https://www.askpython.com/python/list/average-of-a-list-in-python>

在 Python 中有很多方法可以找到一个列表的平均值，只要它们都是相同的类型。在本文中，我们将研究一些方法来寻找一个 [Python 列表](https://www.askpython.com/python/list/python-list)的平均元素。

我们开始吧！

* * *

## 方法 1:使用 reduce()在 Python 中查找列表的平均值

我们可以使用 **reduce** ()方法，以及 lambda 函数(如这里的[所示](https://www.askpython.com/python/python-lambda-anonymous-function#lambda-function-with-reduce))。

我们将使用 lambda 对元素求和，并将结果除以列表的长度，因为 **average =(所有元素的总和)/(元素的数量)**

所以，相应的表达式是:

```py
average = (reduce(lambda x, y: x + y, input_list)) / len(input_list)

```

让我们举一个完整的例子来说明这一点。

```py
from functools import reduce

input_list = [3, 2, 1, 5, 7, 8]

average = (reduce(lambda x, y: x + y, input_list)) / len(input_list)

print("Average of List:", average)

```

**输出**

```py
Average of List: 4.333333333333333

```

这确实是正确的输出，因为平均值是:**(3+2+1+5+7+8)/6 = 26/6 = 4.333**

我们不能传递不同类型的元素，因为`+`操作数可能不起作用。因此，列表中的所有元素都必须与`+`操作数类型兼容，这样才能工作。这是我们可以在 Python 中找到一个列表的平均值的方法之一。

## 方法 2:使用 sum()方法(推荐)

不用`reduce()`的方法，我们可以直接用`sum(list)`求和，除以长度。

```py
input_list = [3, 2, 1, 5, 7, 8]

average = sum(input_list) / len(input_list)

print("Average of List:", average)

```

我们会得到和以前一样的输出！

这是在 Python 中查找列表平均值的推荐方法，因为它只使用内置函数，并且使用起来非常简单。

## 方法 3:使用 statistics.mean()

对于 **3.4** 及以上的 Python 版本，我们也可以使用`statistics`模块在 Python 中求一个列表的平均值。

```py
import statistics

input_list = [3, 2, 1, 5, 7, 8]

average = statistics.mean(input_list)

print("Average of List:", average)

```

## 方法 4:使用 numpy.mean()

如果你想使用基于 **NumPy** 的方法，`numpy.mean()`会帮你做到这一点。

```py
import numpy as np

input_list = [3, 2, 1, 5, 7, 8]

average = np.mean(input_list)

print("Average of List:", average)

```

你也可以将列表转换成一个 **numpy 数组**，然后使用`ndarray.mean`得到平均值。您可以使用对您的用例来说更好的，但是它们的性能是一样的。

```py
import numpy as np

input_list = [3, 2, 1, 5, 7, 8]

# Convert to a numpy ndarray and get the mean
# using ndarray.mean
average = np.array(input_list).mean

print("Average of List:", average)

```

## 方法 5:使用 pandas Series.mean()

如果你用的是`Pandas`，你可以用`pandas.Series(list).mean()`得到一个列表的平均值

```py
import pandas as pd

input_list = [3, 2, 1, 5, 7, 8]

# Convert the list to a Pandas Series using pd.Series(list)
# and then get it's mean using series.mean()
average = pd.Series(input_list).mean()

print("Average of List:", average)

```

**输出**

```py
Average of List: 4.333333333333333

```

* * *

## 结论

在本文中，我们学习了如何使用不同的方法在 Python 中找到列表的平均值。

* * *

## 参考

*   [StackOverflow 问题](https://stackoverflow.com/questions/9039961/finding-the-average-of-a-list)求一个列表的平均值

* * *