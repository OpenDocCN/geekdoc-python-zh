# Python 中如何计算点积？

> 原文：<https://www.askpython.com/python-modules/numpy/dot-product>

你好，初学者！在本文中，我们将看到 python 代码来寻找任何给定量的点积，比如向量或数组。Python 编程语言提供了几种方法来实现这一点，下面将讨论其中一些方法。

***也读作:[Python 中的矢量化——完全指南](https://www.askpython.com/python-modules/numpy/vectorization-numpy)***

## 什么是点积？

首先，让我们了解一下“点积”

在数学中，**点积**(有时称为标量积)是一种代数运算，它从两个等长的数字序列中返回一个值。

这个单一值计算为两个序列中相应元素的乘积之和。这些序列可能是一维向量、多维向量或简单的数字。

让我们举个例子来理解这一点:

假设，两个向量 **A** 和 **B** 是二维数组

A = [ [1 2 ] [3 4] ]，B = [ [5 6] [7 8] ]

然后， **A.B** 给出为

[ [ 19 22] [ 43 50] ]

这计算为[[(1 * 5)+(2 * 7))((1 * 6)+(2 * 8))][((3 * 5)+(4 * 7))((3 * 6)+(4 * 8))]]

## 查找点积的 Python 代码

Python 提供了一种寻找两个序列的点积的有效方法，这就是 numpy 库的 *numpy.dot()* 方法。

Numpy.dot()是一种方法，它将两个序列作为参数，无论是向量还是多维数组，并打印结果，即点积。要使用这个方法，我们必须导入 python 的 [numpy](https://www.askpython.com/python/numpy-linear-algebraic-functions) 库。让我们看几个例子:

### 示例 1:标量的点积

在本例中，我们将采用两个标量值，并使用 numpy.dot()打印它们的点积。

两个标量的点积是通过简单的相乘得到的。

比如说，两个标量 A = 7，B = 6，那么 A.B = 42

```py
#importing numpy library
import numpy as np

#Taking two scalars
a = 3
b = 8

#calculating dot product using dot()
print("The dot product of given scalars = a.b =",np.dot(a,b))

```

上述代码的输出是:

```py
The dot product of given scalars = a.b = 24

```

### 示例 2:数组的点积

这里，我们将采用两个[数组](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)。这些数组可以是一维、二维或多维的。在 dot()的帮助下，我们将计算它们的点积。我们正在考虑点积的两个二维数组。

二维数组的点积通过矩阵乘法来计算。

```py
#importing numpy library
import numpy as np

#Taking two 2-D arrays
a = [ [1, 2], [3, 4]]
b = [ [7, 6], [5, 4]]

#calculating dot product using dot()
print("The dot product of given arrays :")
np.dot(a,b))

```

输出是:

```py
The dot product of given arrays :

array( [ [17, 14],
            [41, 34] ] )

```

***注:***

对于二维或多维数组，点积*不可换。*即 a.b 不等于 b.a 例 2 中，我们把点积算成了 a.b，而不是 b.a，这样会给出完全不同的结果。

### 结论

那么，Python 中计算点积不是很简单吗？有了可用的功能，当然是了。这是我的观点。我希望你理解了这篇文章。更多此类文章，敬请关注 https://www.askpython.com/

在那之前，学习愉快！🙂