# 熊猫数据框架平均值——如何计算平均值？

> 原文：<https://www.askpython.com/python-modules/pandas/pandas-dataframe-mean>

在本文中，我们将计算 Python pandas 中的数据帧平均值。Python 被广泛用于数据分析和处理。所以一般来说 python 是用来处理庞大且未分类的非正式数据的。为了从我们现有的数据中获得有意义的信息，我们使用统计概念，如均值、中值和众数。这些概念有助于我们对数据进行适当的分类和建模，以便提出一个非常有效的模型。

## 什么是卑鄙？

Mean 基本上是我们数据集的平均值。对于一个数据集，算术平均值，也称为算术平均值，是一组有限数字的中心值:具体来说，是值的总和除以值的个数。平均值由以下公式给出:

![A= \frac {1}{n} \sum \limits_{i=1}^n a_i](img/b95c9cd33ac043a3648156916b885101.png)

| ![A](img/39cb941bff0184c97ea436c0d959f033.png) | = | 等差中项 |
| ![n](img/bf9a034cdb2cc8dc1dc9239834526d5d.png) | = | 值的数量 |
| ![a_i](img/255d2802eaa890a6d6dcc3f411785c60.png) | = | 数据集值 |

## 熊猫的数据帧均值

我们在 pandas 中有一个内置的均值函数，可以用在我们的数据框对象上。为了使用 mean 函数，我们需要在代码片段中导入 [pandas 库](https://www.askpython.com/python-modules/pandas/python-pandas-module-tutorial)。现在让我们理解均值函数的基本语法和性质

## 熊猫。数据帧.平均值

mean 函数在应用于系列时将返回系列的平均值，在应用于 dataframe 对象时，将返回 dataframe 中所有系列的平均值列表。现在让我们理解均值函数的语法和参数。

## 句法

DataFrame.mean(axis=None，skipna=None，level=None，numeric_only=None，**kwargs)

## 因素

*   **轴**:取值可以是 0，也可以是 1。默认值为 0，表示索引/行轴。
    当轴= 0 时，该功能应用于索引轴和
*   当 axis = 1 时，它应用于列。
*   **skipna:** 计算结果时排除所有空值。
*   **级别:**与特定级别一起计数，如果轴是多指标(分层)，则折叠成一个系列。
*   **numeric_only:** 只包含 int、float、boolean 列。如果没有，它将尝试使用所有数据，然后只使用数字数据。未针对系列实施。
*   ****kwargs:** 要传递给函数的附加关键字参数。

**返回**系列或数据帧的平均值。

既然我们已经熟悉了函数的语法和参数，现在让我们通过一些例子来理解函数的工作原理。

## 示例–如何计算数据帧平均值

```py
import pandas as pd

data = [[4, 1, 5], [3, 6, 7], [4, 5, 2], [2, 9, 4]]

df = pd.DataFrame(data)

print(df.mean(axis = 0))

```

## 输出

```py
0    3.25
1    5.25
2    4.50
dtype: float64
```

我们可以看到，平均值是为数据帧的每一行/索引计算的

## 示例–用轴 1 计算数据帧平均值

```py
import pandas as pd

data = [[4, 1, 5], [3, 6, 7], [4, 5, 2], [2, 9, 4]]

df = pd.DataFrame(data)

print(df.mean(axis = 1))

```

## 输出

```py
0    3.333333
1    5.333333
2    3.666667
3    5.000000
dtype: float64

```

这里我们可以看到，平均值是为每一列计算的。

在下一个例子中，我们将看到如何将均值函数应用于数据帧中的特定序列。

## 示例 3–计算不带轴的平均值

```py
import pandas as pd

data = [[4, 1, 5], [3, 6, 7], [4, 5, 2], [2, 9, 4]]

df = pd.DataFrame(data)

print(df[0].mean())

```

上面的代码将打印数据帧中第一个索引轴的平均值。

## 输出

```py
3.25
```

这里我们可以验证输出是一个标量值，它是 df[0] = {4，3，4，2}的平均值。即(4+3+4+2)/3 = 3.25

## 结论

通过本文，我们了解了 mean()函数在熊猫图书馆中的用途和应用。

## 参考

[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html)