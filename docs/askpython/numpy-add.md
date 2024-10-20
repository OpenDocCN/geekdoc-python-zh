# NumPy add–以简单的方式解释

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-add>

在本文中，我们将探索 NumPy add 函数。你有没有想过我们如何使用编程将两个数字或两个数字数组相加？这就是 Python NumPy 库发挥作用的地方。

在本教程中，我们将浏览函数的语法，并练习不同类型的例子。那么，我们开始吧。

## 什么是 NumPy add？

**`numpy.add()`** 是 NumPy 库的数学函数之一。它只是将作为输入传递给函数的值相加。

是的，从定义上来说就是这么简单🙂

让我们看看语法，了解更多关于函数的内容。

### NumPy add 的语法

```py
numpy.add(x1 , x2)

```

这里输入的**`x1`****`x2`**是 NumPy 数组的数字。

**注意:**在语法中，输入也可以是标量值(简单数字)和 Python 列表。

## 使用 NumPy add

我们准备做编程，多了解一下这个功能。

### NumPy 与标量值相加

```py
# Importing the NumPy module
import numpy as np

a = 5
b = 4

print("The sum of 5 and 4 is:",np.add(a , b))

```

**输出**

```py
The sum of 5 and 4 is: 9

```

我们首先使用上面代码片段中的 **`import`** 语句导入 NumPy 库。在下面几行中，我们给变量 **`a`** 和 **`b`** 赋值。

为了计算 **`a`** 和 **`b`** 的值之和，调用函数 **`np.add(a , b)`** 。

现在让我们使用带有 NumPy 数组和一个标量值的函数，并观察输出。

### NumPy 添加一个 NumPy 数组和一个标量值

```py
import numpy as np

# Creating the  2-D array
a = np.array([[2 , 4 , 7] , [5 , 10 , 15]])

b = 5

c = np.add(a , b)

print("Input Array:\n",a)
print("After adding 5 to each value of the Input Array:\n",c)

```

**输出**

```py
Input Array:
 [[ 2  4  7]
 [ 5 10 15]]
After adding 5 to each value of the Input Array:
 [[ 7  9 12]
 [10 15 20]]

```

在上面的代码片段中，使用 **`np.array()`** 创建了一个二维数组，该数组存储在变量 **`a`** 中。

接下来，我们创建了一个保存整数值 5 的变量 **`b`** 。进一步，我们使用 **`np.add()`** ，其中我们传递了 **`a`** 和 **`b`** 作为函数的参数，函数的输出存储在变量 **`c`** 中。

在输出中，我们得到了一个二维数组。如果您仔细观察输出数组，您会看到数字 5 被添加到输入数组中的每个元素。

**`Note:`** 输出数组的形状与输入数组的形状相同。

### NumPy 添加两个大小相同的 NumPy 数组

在本例中，我们将添加两个大小相同的 numpy 阵列。

```py
import numpy as np

# Creating 2x4 array
a = np.array([[2 , 5 , 7 , 3] , [1 , 4 , 5 , 12]])

b = np.array([[9 , 5 , 11 , 23] , [21 , 34 , 1 , 9]])

# Using the add function
c = np.add(a , b)

# Printing the values
print("Array 1:\n",a)
print("Array 2:\n",b)

print("Output array:\n",c)

```

**输出**

```py
Array 1:
 [[ 2  5  7  3]
 [ 1  4  5 12]]
Array 2:
 [[ 9  5 11 23]
 [21 34  1  9]]
Output array:
 [[11 10 18 26]
 [22 38  6 21]]

```

在上面的例子中，我们创建了两个 2×4 维的 numpy 数组，即一个 2 行 4 列的数组。这些数组存储在变量`a`和 **`b`** 中。

在下面几行中，我们使用了函数 **`np.add(a , b)`** ，其中 a 和 b 作为参数传递给函数。函数 **`np.add(a,b)`** 将数组 **`a`** 和 **`b`** 中相同位置的值相加。

输出数组的**形状**与输入数组相同。现在，让我们看看文章中最有趣的例子。

### NumPy 加法器具有两个不同形状 NumPy 阵列

在这个例子中，我们将看到如何将一个 1 维数字数组添加到一个 2 维数字数组中。

```py
import numpy as np

# Creating a 1-D array
a = np.array((10 , 56 , 21))

# Creating a 2-D array
b = np.array([[1 , 4 , 6] , [23 , 12 , 16]])

c = np.add(a , b)

print("Array 1:\n",a)
print("Array 2:\n",b)

print("Output Array:\n",c)

```

**输出**

```py
Array 1:
 [10 56 21]
Array 2:
 [[ 1  4  6]
 [23 12 16]]
Output Array:
 [[11 60 27]
 [33 68 37]]

```

添加两个不同形状的 NumPy 数组不是很有趣吗？嗯，我们使用 NumPy 库轻松地做到了这一点。

让我们试着理解上面的片段。我们创建了一个存储在变量 **`a`** 中的 **1-D** NumPy 数组，另一个 2-D NumPy 数组被创建并存储在变量 **`b`** 中。

如果您仔细观察以上示例中创建的数组，您会注意到它们具有相同的列数，但行数不同。这就是术语**广播**发挥作用的地方。

这里，函数 **`np.add()`** 在二维数组的行之间传播一维数组。在上面的例子中， **`np.add()`** 将数组 **`a`** 的值添加到数组 **`b`** 的 row1 中，以元素为单位。类似地，对数组 **b** 的下一行进行相同的计算。

**注意:**当一维数组中的元素数与二维数组中的列数相同时，上述情况是可能的

所以，我们完成了例子🙂

## 摘要

在本文中，我们学习了如何对各种输入类型使用 NumPy add 函数。这是一个使用起来很简单的函数，因为它只是将输入添加到函数中。尝试使用 NumPy add 函数和 NumPy 数组的不同函数。坚持学习，快乐编码。点击了解更多文章[。](https://www.askpython.com/)