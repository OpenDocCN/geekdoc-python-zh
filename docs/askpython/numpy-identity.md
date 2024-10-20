# NumPy identity 函数:返回一个主对角线上有 1 的正方形数组

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-identity>

在本文中，我们将尝试理解 Python 中 NumPy 的 identity 函数。

Python 包 NumPy(数值 Python)用于操作数组。使用 NumPy 可以在一个数组上执行许多数学运算。

它提供了一个庞大的高级数学函数库来处理这些数组和矩阵，并提供了强大的数据结构来确保使用数组和矩阵进行高效计算。

2005 年，特拉维斯·奥列芬特开发了 NumPy。您可以免费使用它，因为它是一个开源项目。

***也读作: [Numpy 渐变:返回 N 维数组的渐变](https://www.askpython.com/python/numpy-gradient)***

## numpy.identity()是什么？

该函数用于返回一个主对角线上有 1 的正方形数组(行数和列数相等的数组)，这种数组称为*恒等数组*。

## numpy.identity()的语法

```py
numpy.identity(n, dtype=None, like=None)

```

### 因素

*   **n: int**
    *   需要
    *   所需输出数组的行数或列数
*   **dtype:数据类型**
    *   可选择的
    *   输出数组中值的数据类型，
    *   默认设置为浮动。
*   **like: array_like**
    *   可选择的
    *   可以使用引用对象创建非 NumPy 数组。如果符合**数组函数**协议，则结果将由 as like 中提供的数组 like 确定。在这种情况下，它确保创建的数组对象与作为参数提供的对象兼容。

返回一个 n x n 对称数组，其主对角线设置为 1，所有剩余元素设置为零。

## numpy.identity()的实现

在使用这个函数之前，请确保在 IDE 中导入 NumPy 包。要导入 NumPy 包，请运行以下代码行。

```py
import numpy as np

```

### 示例 1:只传递“n”参数

```py
np.identity(4)

```

**输出**

![Screenshot 694](img/e8b3b0b409319423e00cfa00b72116f5.png)

Example 1

### 示例 2:传递其他参数

```py
np.identity(3, dtype=int)

np.identity(3, dtype=complex)

```

**输出**

![Example 2](img/55f5005a204daec90ec718342d825a15.png)

Example 2

## 摘要

通过使用 Python 中的 NumPy 包，处理数组变得很容易。identity()函数是创建一个 n×n 单位矩阵/数组的简单方法。

## 参考

[https://numpy . org/doc/stable/reference/generated/numpy . identity . html](https://numpy.org/doc/stable/reference/generated/numpy.identity.html)