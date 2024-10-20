# NumPy 乘法——以简单的方式说明

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-multiply>

嘿大家好！欢迎来到另一个关于 NumPy 函数的教程。在本教程中，我们将详细探讨 NumPy 乘法函数。

我们在日常生活中都会做乘法运算。无论是我们的数学课还是店主为了记录商店里的商品而做的计算。今天我们将看到如何使用 python 编程将两个数字或两个数字数组相乘。

让我们开始吧。

***也读作:[NumPy linalg . det——计算给定数组的行列式](https://www.askpython.com/python-modules/numpy/numpy-linalg-det)***

## 什么是 NumPy 乘法？

NumPy multiply 是 NumPy 库的数学函数之一，它将传递给函数的输入相乘。

让我们来看看函数的语法。

### NumPy 乘法语法

```py
numpy.multiply(x1 , x2)

```

这里，输入 **`x1`** 和 **`x2`** 可以是标量数字，也可以是数字的 NumPy 数组。

## 使用 NumPy 乘法

我们来做一些 python 编程吧。

### 与标量值相乘

```py
# Importing the NumPy module
import numpy as np

a = 5
b = 4

print("The product of 5 and 4 is:",np.multiply(a , b))

```

**输出**

```py
The product of 5 and 4 is: 20

```

我们首先使用上面代码片段中的 **`import`** 语句导入 NumPy 库。在 **`print`** 语句中，调用函数 **`np.multiply(a,b)`** ，其中 a 和 b 作为输入传递给函数。

在这个代码片段中，输出也是一个**标量**值。

### NumPy 乘以一个 NumPy 数组和一个标量值

```py
# Importing the NumPy module
import numpy as np

# Creating the  2-D array
a = np.array([[2 , 4 , 7] , [5 , 10 , 15]])

b = 10

c = np.multiply(a , b)

print("Input Array:\n",a)
print("After multiplying 10 to each value of the Input Array:\n",c)

```

**输出**

```py
Input Array:
 [[ 2  4  7]
 [ 5 10 15]]
After multiplying 10 to each value of the Input Array:
 [[ 20  40  70]
 [ 50 100 150]]

```

在上面的例子中，使用函数 **`np.array()`** 创建了一个大小为 2×3 的二维数组。在下面几行中，通过将 **`a`** 和 **`b`** 作为参数传递给函数来调用函数 **`np.multiply(a,b)`** ，其中 **`a`** 是 NumPy 数组， **`b`** 保存标量值 10。

在输出中，函数 **`np.multiply(a,b)`** 将 NumPy 数组的所有值乘以 10。

### NumPy 乘以两个相同大小的 NumPy 数组

```py
import numpy as np

# Creating 2x2 array
a = np.array([[2 , 5] , [1 , 4]])

b = np.array([[9 , 5] , [21 , 34]])

# Using the multiply function
c = np.multiply(a , b)

# Printing the values
print("Array 1:\n",a)
print("Array 2:\n",b)

print("Output array:\n",c)

```

**输出**

```py
Array 1:
 [[2 5]
 [1 4]]
Array 2:
 [[ 9  5]
 [21 34]]
Output array:
 [[ 18  25]
 [ 21 136]]

```

在本例中，使用函数 **`np.array()`** 创建了两个大小为 2×2 的 NumPy 数组，并存储在变量 **`a`** 和 **`b`** 中。接下来，通过将 **`a`** 和 **`b`** 作为参数来调用函数`np.multiply(a,b)`，其中 **`a`** 和 **`b`** 是我们之前使用函数`np.array()`创建的 NumPy 数组。

在输出中，数组包含两个输入数组中相同位置处**的值的乘积。**

**注意:**输出数组与输入数组大小相同。

### 矩阵和向量相乘

```py
import numpy as np

# Creating a vector or 1-D array
a = np.array((10 , 20 , 30))

# Creating a matrix or 2-D array
b = np.array([[1 , 2 , 4] , [8 , 10 , 16]])

c = np.multiply(a , b)

print("Array 1:\n",a)
print("Array 2:\n",b)

print("Output Array:\n",c)

```

**输出**

```py
Array 1:
 [10 20 30]
Array 2:
 [[ 1  2  4]
 [ 8 10 16]]
Output Array:
 [[ 10  40 120]
 [ 80 200 480]]

```

这是最有趣的例子。这里，我们创建了一个具有 3 个元素的向量或一维数组，以及一个大小为 2×3 的二维数组或矩阵，即具有 2 行 3 列。在下面几行中，通过传递 **`a`** 和 **`b`** 作为参数来调用函数 **`np.multiply(a,b)`** ，其中 a 是**向量**，b 是**矩阵**。

在这种情况下，NumPy 执行**广播**。它获取向量并将其与矩阵中的每一行相乘。这是可能的，因为向量的元素数与矩阵中的列数相同。

在输出数组中，第**行第**个元素是通过将向量乘以矩阵的第一行得到的，第**行第二**个元素是通过将向量乘以矩阵的第二行得到的。

至此，我们完成了本教程的所有示例。您应该尝试将该函数用于您选择的示例，并观察输出。

## 摘要

在本文中，我们学习了 NumPy 乘法函数，并练习了不同类型的示例。这真的是一个简单易用且易于理解的函数。继续探索更多这样的教程[这里](http://askpython.com)。

## 参考

[NumPy 文档–NumPy 乘数](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)