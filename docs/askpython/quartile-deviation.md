# 四分位数偏差——在 Python 中实现

> 原文：<https://www.askpython.com/python/examples/quartile-deviation>

嘿伙计们！在本教程中，我将向您展示如何使用 Python 编程语言计算**四分位数偏差**。

离差的绝对度量被称为**四分位数偏差**。它的计算方法是将上下四分位数之差除以一半。

***也读:[绘制数学函数——如何用 Python 绘制数学函数？](https://www.askpython.com/python/examples/plot-mathematical-functions)***

* * *

## 四分位偏差介绍

四分位偏差是离差的绝对度量，其中离差是分布值与平均值之间的差异量。

即使数据中只有一个极高或极低的数字，该范围作为离差度量的效用也会降低。

为了计算四分位数偏差，我们必须将数据分成四个部分，每个部分包含 25%的值。

数据的四分位数偏差通过取最高(75%)和最低(25%)四分位数之差的一半来计算。

* * *

## 在 Python 中实现四分位数偏差

我希望你现在明白什么是四分位偏差。让我们看看如何使用 Python 来确定数据集的四分位数偏差。

为了在 Python 中计算它，我们将首先构建一个数据集，然后从数据中识别 quartile1、quartile2 和 quartile3，然后开发一个函数，该函数将用于返回 quartile3 和 quartile1 之差的一半的乘积。

看看下面提到的代码:

```py
import numpy as np
data = list(range(20, 100, 5))
print("Initial Data : ", data)

Q1 = np.quantile(data, 0.25)
Q2 = np.quantile(data, 0.50)
Q3 = np.quantile(data, 0.75)

print("Quartile 1 : ", Q1)
print("Quartile 2 : ", Q2)
print("Quartile 3 : ", Q3)

def QuartileDeviation(a, b):
    return (a - b)/2
print("Computed Result : ",QuartileDeviation(Q3, Q1))

```

* * *

## 代码的输出

上面提到的代码将给出以下输出:

```py
Initial Data :  [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Quartile 1 :  38.75
Quartile 2 :  57.5
Quartile 3 :  76.25
Computed Result :  18.75

```

* * *

我希望你喜欢这篇关于用 Python 编程语言计算数据集的四分位数偏差的教程。

多看看这样的教程，永远不要停止学习！

1.  [Numpy vstack()方法–完整概述](https://www.askpython.com/python-modules/numpy-vstack)
2.  [将 Pandas 数据帧转换为 Numpy 数组【分步】](https://www.askpython.com/python-modules/numpy/pandas-dataframe-to-numpy-array)
3.  [NumPy 中的 3 个简单排序技巧](https://www.askpython.com/python/sorting-techniques-in-numpy)
4.  [要了解的 5 种数据分布](https://www.askpython.com/python-modules/numpy/numpy-data-distributions)

* * *