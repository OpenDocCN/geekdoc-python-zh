# Numpy 数组的内积——快速指南

> 原文：<https://www.askpython.com/python-modules/numpy/inner-product-numpy-array>

在本文中，我们将学习如何在两个数组之间执行内积运算。我们将研究一维数组和多维数组。让我们先来看看什么是 Numpy 数组。

## 什么是 NumPy 数组？

Numpy 是一个用于科学计算的开源 python 库。Numpy 数组类似于列表，除了它包含相似数据类型的对象，并且比列表快得多。

对于科学计算，它们是 Python 中最重要的数据结构之一。numpy 数组高效、通用且易于使用。它们也是多维的，这意味着它们可以在多个维度上存储数据。维数称为数组的秩。数组可以有任何秩，但大多数数组要么有一维，要么有二维。

让我们看看如何创建一个 Numpy 数组。

```py
import numpy as np
a=np.array([1,2,3])
print (a)

```

输出

```py
[1 2 3]

```

## Numpy 阵列上的内积

我们可以借助一个简单的 numpy.inner()函数来执行数组的内积。

语法:

```py
numpy.inner(arr1, arr2)=sum(array1[:] , array2[:])

```

## 一维 Numpy 数组的内积

您可以对 Numpy 数组的一维内积使用以下代码。

```py
import numpy as np 
a= np.array([1,2,3])
b= np.array([0,1,0])
product=np.inner(a,b) 
print(product)

```

**输出**

```py
2

```

这里的输出乘积等于[1*0+2*1+3*0]=2

## 多维数组的内积

您可以对多维数组使用以下代码。

```py
import numpy as np 
a = np.array([[1,3], [4,5]]) 
b = np.array([[11, 12], [15, 16]]) 

product=np.inner(a,b)
print(product)

```

输出

```py
[[ 47  63]
 [104 140]]

```

## 结论

总之，我们学习了如何在 Numpy 数组上执行内积。希望这篇文章对你有用！