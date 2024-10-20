# Python numpy.reshape()函数

> 原文：<https://www.askpython.com/python-modules/numpy/python-numpy-reshape-function>

嘿，伙计们！希望你们都过得好。在本文中，我们将了解**Python numpy . shape()函数**的工作原理。

众所周知， [Python NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)为我们提供了海量的函数来操作和处理数学数据。在这里，我们将揭示 Numpy reshape()函数的功能。

所以，让我们开始吧！

* * *

## Python numpy.reshape()函数的工作原理

`Python numpy.reshape() function`使我们能够改变[数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)的形状，即改变数组元素的尺寸。重塑一个数组将有助于我们改变驻留在特定维度中的数据值的数量。

需要注意的重要一点是， **reshape()函数保留了数组**的大小，也就是说，它没有改变数组元素的数量。

现在让我们在下一节中了解 numpy.reshape()函数的结构。

* * *

## Python numpy.reshape()函数的语法

```py
array-name.reshape(shape)

```

*   `shape`:是整数值的元组，根据这个元组对元素进行整形。

让我们考虑一个例子来理解将形状传递给 shape()函数的过程。

如果输入数组中有 16 个元素，那么我们需要将这些整数值作为元组传递给 shape 参数，这些元组值的乘积等于元素的数量，即 16。

形状参数可能有以下几种情况:

*   [2,8]
*   [8,2]
*   [4,4]
*   [16,1]
*   [1,16]
*   [4,2,2]

现在让我们通过下面的例子来更好地理解 numpy.reshape()函数。

* * *

## 用示例实现 Python numpy.reshape()

在下面的例子中，我们使用 numpy.arange()函数创建了一个包含 16 个元素的一维数组。

此外，我们使用 shape()函数将数组的维度重新调整为一个二维数组，每个维度有 4 个元素。

```py
import numpy as np 

arr = np.arange(16) 
print("Array elements: \n", arr) 

res = np.arange(16).reshape(4, 4) 
print("\nArray reshaped as 4 rows and 4 columns: \n", res) 

```

**输出:**

```py
Array elements: 
 [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

Array reshaped as 4 rows and 4 columns: 
 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]

```

现在，我们已经改变了 1 维数组的形状，并将其转换为每维 2 个元素的数组。

```py
import numpy as np 

arr = np.arange(16) 
print("Array elements: \n", arr) 

res = np.arange(16).reshape(4,2,2) 
print("\nArray reshaped: \n", res) 

```

**输出:**

```py
Array elements: 
 [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

Array reshaped: 
 [[[ 0  1]
  [ 2  3]]

 [[ 4  5]
  [ 6  7]]

 [[ 8  9]
  [10 11]]

 [[12 13]
  [14 15]]]

```

在下面的示例中，我们通过将值-1 传递给 shape()函数，将二维数组转换为一维数组。

```py
import numpy as np 

arr = np.array([[1, 2, 3,4], [10, 11, 12,13],[5,6,7,8]])
print("Array elements: \n", arr) 

res = arr.reshape(-1) 
print("\nArray reshaped as 1-D Array: \n", res) 

```

**输出:**

```py
Array elements: 
 [[ 1  2  3  4]
 [10 11 12 13]
 [ 5  6  7  8]]

Array reshaped as 1-D Array: 
 [ 1  2  3  4 10 11 12 13  5  6  7  8]

```

* * *

## 结论

到此，我们就结束了这个话题。如果你有任何疑问，欢迎在下面评论。快乐学习！

* * *

## 参考

*   [NumPy reshape()函数—文档](https://numpy.org/doc/1.18/reference/generated/numpy.reshape.html)