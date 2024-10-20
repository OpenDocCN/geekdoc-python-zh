# Numpy vstack()方法–完整概述

> 原文：<https://www.askpython.com/python-modules/numpy-vstack>

大家好！在本教程中，我们将学习什么是 Numpy `vstack()`方法，以及如何在 Python 中使用它。所以让我们开始吧。

## numpy.vstack()方法是什么？

`Numpy.vstack()`是 Python 中的一个函数，它接受一个数组的[元组](https://www.askpython.com/python/tuple/python-tuple)，并沿第一维垂直连接它们，使它们成为一个数组。

它的语法是:

```py
numpy.vstack(tup)

```

它的参数是一个 tuple，这是一个我们想要连接的 n 数组序列。除了第一个轴，数组在所有轴上都必须具有相同的形状。

该方法返回一个 ndarray，它是通过堆叠输入中给定的数组形成的。返回的数组至少有二维。

## Numpy vstack()的示例

对于线性 1-D 阵列，所有阵列垂直堆叠以形成 2-D 阵列。所有输入数组的长度必须相同。

```py
import numpy

a = numpy.array([1, 2, 3, 4, 5])
b = numpy.array([6, 7, 8, 9, 10])
c = numpy.array([11, 12, 13, 14, 15])

print("Shape of array A:", a.shape)
print("Shape of array B:", b.shape)
print("Shape of array C:", c.shape)
print()

stack = numpy.vstack((a, b, c))
print("Shape of new stacked array:", stack.shape)
print("Stacked array is")
print(stack)

```

```py
Shape of array A: (5,)
Shape of array B: (5,)
Shape of array C: (5,)

Shape of new stacked array: (3, 5)
Stacked array is
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]]

```

对于 N 维数组，数组沿第一维堆叠，如下例所示。

```py
import numpy

a = numpy.array([ [1, 2, 3], [4, 5, 6] ])
b = numpy.array([ [7, 8, 9], [10, 11, 12] ])

print("Shape of array A:", a.shape)
print("Shape of array B:", b.shape)
print()

stack = numpy.vstack((a, b))
print("Shape of new stacked array:", stack.shape)
print("Array is")
print(stack)

```

**输出:**

```py
Shape of array A: (2, 3)
Shape of array B: (2, 3)

Shape of new stacked array: (4, 3)
Array is
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]]

```

对于 N 维数组，除了如下所示的第一维，数组的形状在所有维度上都必须相同。

```py
import numpy

a = numpy.array([ [1, 2], [3, 4] ])
b = numpy.array([ [5, 6], [7, 8], [9, 10] ])

print("Shape of array A:", a.shape)
print("Shape of array B:", b.shape)
print()

stack = numpy.vstack((a, b))
print("Shape of new stacked array:", stack.shape)
print("Array is")
print(stack)

```

```py
Shape of array A: (2, 2)
Shape of array B: (3, 2)

Shape of new stacked array: (5, 2)
Array is
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]]

```

## 结论

在这个 Python 教程中，我们学习了 NumPy 模块中的`vstack()`方法。这个函数对于多达 3 维的数组最有意义。例如，对于具有高度(第一轴)、宽度(第二轴)和 r/g/b 通道(第三轴)的像素数据。

感谢阅读！！