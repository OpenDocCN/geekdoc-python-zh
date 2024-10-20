# python——NumPy 数组简介

> 原文：<https://www.askpython.com/python-modules/numpy/python-numpy-arrays>

NumPy 是最常用的科学计算 Python 库。它提供了一个快速的 Pythonic 接口，同时仍然使用更快的 C++进行计算。这确保了高级可读性和 Python 特性仍然存在，同时使实际计算比纯 Python 代码快得多。

这里，我们看看 NumPy 完成所有工作背后的数据结构，以及我们如何以不同的方式转换它，就像我们如何操作其他类似数组的数据结构一样。

## NumPy 数组对象

为了声明一个 numpy 数组对象，我们首先导入`numpy`库，然后使用`np.array()`库函数实例化我们新创建的数组。

下面的代码片段声明了一个简单的一维 numpy 数组:

```py
>>> import numpy as np
>>> a = np.array([1, 2, 3, 4])
>>> print(a)
[1 2 3 4]

```

每个数组都有以下属性:

*   `ndim`(维数)
*   `shape`(每个尺寸的大小)
*   `size`(数组的总大小)
*   `dtype`(数组的数据类型)

NumPy 数组元素有相同的数据类型，不像 [Python 列表](https://www.askpython.com/python/list/python-list)。因此，我们不能让单个 numpy 数组保存多种不同的数据类型。

要声明一个更高维的数组，类似于用任何其他语言声明一个更高维的数组，使用适当的矩阵来表示整个数组。

```py
# Declare a 2-Dimensional numpy array
b = np.array([[1, 2, 3], [4, 5, 6]])
print("b -> ndim:", b.ndim)
print("b -> shape:", b.shape)
print("b -> size:", b.size)
print("b -> dtype:", b.dtype)

```

输出:

```py
b -> ndim: 2
b -> shape: (2, 3)
b -> size: 6
b -> dtype: dtype('int64')

```

* * *

## 访问 NumPy 数组元素

与 Python 中访问列表元素和数组元素类似，numpy 数组的访问方式也是一样的。

为了访问多维数组中的单个元素，我们对每个维度使用逗号分隔的索引。

```py
>>> b[0]
array([1, 2, 3])
>>> b[1]
array([4, 5, 6])
>>> b[-1]
array([4, 5, 6])
>>> b[1, 1]
5

```

* * *

## NumPy 数组切片

再次，类似于 Python 标准库，numpy 也为我们提供了对 NumPy 数组的切片操作，使用它我们可以访问元素的数组切片，给我们一个对应的子数组。

```py
>>> b[:]
array([[1, 2, 3],
       [4, 5, 6]])
>>> b[:1]
array([1, 2, 3])

```

事实上，由于 numpy 操作的高度优化特性，这是广泛推荐的使用 NumPy 数组的方式。因为相比之下，原生 python 方法非常慢，所以我们应该只使用 numpy 方法来操作 numpy 数组。结果，纯 Python 迭代循环和其他列表理解不能与 numpy 一起使用。

* * *

## 生成 numpy 数组的其他方法

我们可以使用 numpy 内置的`arange(n)`方法来构造一个由数字`0`到`n-1`组成的一维数组。

```py
>>> c = np.arange(12)
>>> print(c)
[0 1 2 3 4 5 6 7 8 9 10 11]
>>> c.shape
(12,)

```

使用`random.randint(limit, size=N)`生成一个随机整数数组，所有元素在 0 和`limit`之间，大小为`N`，指定为关键字参数。

```py
>>> d = np.random.randint(10, size=6)
>>> d
array([7, 7, 8, 8, 3, 3])
>>> e = np.random.randint(10, size=(3,4))
>>> e
array([[2, 2, 0, 5],
       [8, 9, 7, 3],
       [5, 7, 7, 0]])

```

* * *

## 操作 NumPy 数组

numpy 提供了一个方法`reshape()`，可以用来改变 NumPy 数组的维度，并就地修改原来的数组。这里，我们展示了一个使用`reshape()`将`c`的形状改变为`(4, 3)`的例子

```py
>>> c
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> c.shape
(12,)
>>> c.reshape(4, 3)
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])

```

因为 numpy 操作被设计为高度优化的，所以从数组创建的任何子数组仍然保存对原始数组的引用。这意味着如果子数组被就地修改，原始数组也会被修改。

```py
>>> f = e[:3, :2]
>>> f
array([[2, 2],
       [8, 9],
       [5, 7]])
>>> f[0,0] *= 3
>>> f
array([[6, 2],
       [8, 9],
       [5, 7]])
>>> e
array([[6, 2, 0, 5],
       [8, 9, 7, 3],
       [5, 7, 7, 0]])

```

这里，原始数组`e`也随着子数组片`f`的任何变化而被修改。这是因为 numpy 切片只返回原始数组的一个**视图**。

为了确保原始数组不随子数组片的任何变化而修改，我们使用 numpy `copy()`方法创建数组的副本并修改克隆对象，而不是处理对原始对象的引用。

下面的片段展示了`copy`如何处理这个问题。

```py
>>> e
array([[6, 2, 0, 5],
       [8, 9, 7, 3],
       [5, 7, 7, 0]])
>>> f = e[:3, :2].copy()
>>> f
array([[6, 2],
       [8, 9],
       [5, 7]])
>>> f[0,0] = 100
>>> f
array([[100,   2],
       [  8,   9],
       [  5,   7]])
>>> e
# No change is reflected in the original array
# We are safe!
array([[6, 2, 0, 5],
       [8, 9, 7, 3],
       [5, 7, 7, 0]])

```

* * *

## 结论

在本文中，我们学习了 numpy 数组和一些涉及它们的基本操作，包括它们的属性、数组切片、整形和复制。

## 参考

[数量文件](https://numpy.org/doc/)