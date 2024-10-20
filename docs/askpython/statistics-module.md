# Python 统计模块–需要了解的 7 个函数！

> 原文：<https://www.askpython.com/python-modules/statistics-module>

Python 统计模块提供了对给定的一组数字计算数学统计数据的函数。它是在 Python 3.4 版本中引入的。这是一个非常简单的模块，可以处理整数、浮点数、小数和分数。在本文中，我们将关注 Python 统计模块的 **7 个重要函数。**

* * *

## Python 统计模块函数

我们将关注 Python 中统计模块提供的一些最突出的功能。

*   **均值()函数**
*   **中值()函数**
*   **median_high()函数**
*   **median_low()函数**
*   **stdev()函数**
*   **_sum()函数**
*   **_counts()函数**

让我们一个一个来看看。

* * *

## 1.mean()函数

[均值](https://www.askpython.com/python/examples/mean-and-standard-deviation-python)是最常用的统计量之一，可以一目了然地了解数据。平均值代表一次全部数据的总体平均估计值。它的计算方法是将数据集中的所有值相加，然后除以值的个数。

例如，如果数据集是[1，2，3，4，5]，则平均值将是(1+2+3+4+5)/5 = 3。

`statistics.mean()`函数返回一组数字数据值的平均值。

**语法:**

```py
statistics.mean(data)

```

* * *

## 2.median()函数

除了平均值之外，我们经常会遇到这样的情况:我们需要一个值来表示整个数据的中间部分。使用`statistics.median()`函数，我们可以计算数据值的中间值。中值是在从最小值到最大值对数据集进行排序后得出的。如果数据集有偶数个值，则中位数是中间两个数的平均值。

例如，如果数据集是[1，3，10，2]，那么首先我们将按升序排列它，即[1，2，3，10]。因为有偶数个值，所以中值将是中间两个数字(即 2 和 3)的平均值。所以中位数是 2.5。对于数据集[1，10，3]，中值将为 3。

**语法:**

```py
statistics.median(data)

```

* * *

## 3.median_high()函数

统计模块的`median_high()`函数返回数据集中较高的中值。当数据值本质上是离散的时，高中值特别有用。如果数据集有偶数个值，则返回中间两个值中较高的一个。对于奇数个值，median_high 与中值相同。

例如，如果数据集是[1，2，3，10]，则 median_high 将是 3。如果数据集为[1，3，5]，则 median_high 与中值 3 相同。

**语法:**

```py
statistics.median_high(data)

```

* * *

## 4.statistics.median_low()函数

`median_low()`函数返回一组值中的最低中值。当数据本质上是离散的，并且我们需要精确的数据点而不是插值点时，这是很有用的。如果数据集有偶数个值，则返回中间两个值中较小的一个。对于奇数个值，median_low 与中值相同。

例如，如果数据集是[1，2，3，10]，则 median_low 将是 2。如果数据集是[1，3，5]，则 median_low 与中值 3 相同。

**语法:**

```py
statistics.median_low(data)

```

* * *

## 5.statistics.stdev 函数

`stdev()`函数返回数据的标准偏差。首先，计算数据的平均值。然后计算变化量。方差的平方根就是数据集的标准差。

**语法:**

```py
statistics.stdev(data)

```

* * *

## 6.统计的 _sum()函数

当涉及到作为参数传递的数据点的累积时，就需要使用 _sum()函数了。使用`_sum()`函数，我们可以得到所有数据值的总和以及传递给它的所有数据点的计数。

**语法:**

```py
statistics._sum(data)

```

* * *

## 7.counts()函数

使用`_counts()`函数，我们可以从该组值中获得每个数据点的频率。它计算每个数据点的出现次数，并返回大小为 2 的元组列表。元组的第一个值是数据集值，第二个值是出现次数。

* * *

## Python 统计模块函数示例

让我们看一些使用统计模块函数的例子。

```py
import statistics

data = [10, 203, 20, 30, 40, 50, 60, 70, 80, 100]
res = statistics.mean(data)
print("Mean: ", res)

res = statistics.median(data)
print("Median: ", res)

res = statistics.median_high(data)
print("Median High value: ", res)

res = statistics.median_low(data)
print("Median Low value: ", res)

res = statistics.stdev(data)
print("Standard Deviation: ", res)

res = statistics._sum(data)
print("Sum: ", res)

res = statistics._counts(data)
print("Count: ", res)

```

**输出:**

```py
Mean:  66.3
Median:  55.0
Median High value:  60
Median Low value:  50
Standard Deviation:  55.429735301150004
Sum:  (<class 'int'>, Fraction(663, 1), 10)
Count:  [(10, 1), (203, 1), (20, 1), (30, 1), (40, 1), (50, 1), (60, 1), (70, 1), (80, 1), (100, 1)]    

```

* * *

## 摘要

Python 统计模块对于获取数值数据集的平均值、中值、众数和标准差非常有用。他们处理数字，并提供简单的函数来计算这些值。但是，如果您已经在使用 [NumPy](https://www.askpython.com/python-modules/numpy/python-numpy-module) 或 [Pandas](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial) 模块，您可以使用它们的函数来计算这些值。

## 下一步是什么？

*   [SciPy 模块](https://www.askpython.com/python-modules/python-scipy)
*   [NumPy 的意思是](https://www.askpython.com/python-modules/numpy/mean-of-a-numpy-array)
*   [Python 中计算 SD 的 3 种方法](https://www.askpython.com/python/examples/standard-deviation)

## 资源

*   [Python.org 文件](https://docs.python.org/3/library/statistics.html#module-statistics)
*   [numpy.org 平均文件数](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)