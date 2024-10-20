# NumPy 线性代数函数要知道！

> 原文：<https://www.askpython.com/python/numpy-linear-algebraic-functions>

读者朋友们，你们好！在本文中，我们将关注 Python 中的 **NumPy 线性代数函数**。所以，让我们开始吧！🙂

NumPy 模块为我们提供了各种处理和操作数据的功能。它使我们能够在数组数据结构中创建和存储数据。接下来，它为我们提供了各种函数来分析和操作数据值。

## NumPy 线性代数函数列表

### 1。NumPy 模块提供的矩阵函数

通过 NumPy 模块，我们可以在[数组结构](https://www.askpython.com/python/array/python-array-declaration)上执行线性代数矩阵函数。

在本主题课程中，我们将了解以下功能

1.  **矩阵的秩**:我们可以使用 numpy.linalg.matrix_rank()函数计算数组的秩。
2.  **行列式**:numpy . linalg . det()函数帮助我们把数组当作一个矩阵来计算它的行列式。
3.  **求逆**:inv()函数使我们能够计算数组的逆。
4.  **指数**:使用 numpy.linalg.matrix_power()函数，我们可以对矩阵取幂值，并获取结果。

**举例:**

在下面的例子中，我们使用 numpy.array()函数创建了一个数组。此外，我们对阵列执行了上述线性代数运算，并打印了结果。

```py
import numpy

x = numpy.array([ [2, 8, 7],
                 [6, 1, 1],
                [4, -2, 5]])

print("Rank: ", numpy.linalg.matrix_rank(x))
det_mat = numpy.linalg.det(x) 
print("\nDeterminant: ",det_mat)
inv_mat = numpy.linalg.inv(x)
print("\nInverse: ",inv_mat) 
print("\nMatrix raised to power y:\n",
           numpy.linalg.matrix_power(x, 8))

```

**输出:**

```py
Rank:  3

Determinant:  -306.0

Inverse:  [[-0.02287582  0.17647059 -0.00326797]
 [ 0.08496732  0.05882353 -0.13071895]
 [ 0.05228758 -0.11764706  0.1503268 ]]

Matrix raised to power y:
 [[ 85469036  43167250 109762515]
 [ 54010090  32700701  75149010]
 [ 37996120  22779200  52792281]]

```

* * *

### 2。数字阵列特征值

NumPy 线性代数函数有 linalg 类，它有 **eigh()函数**来计算传递给它的数组元素的特征值。

看看下面的语法！

**语法:**

```py
numpy.linalg.eigh(array)

```

eigh()函数返回一个复矩阵或实对称矩阵的特征值和特征向量。

**举例:**

```py
from numpy import linalg as li

x = numpy.array([[2, -4j], [-2j, 4]])

res = li.eigh(x)

print("Eigen value:", res)

```

**输出:**

```py
Eigen value: (array([0.76393202, 5.23606798]), array([[-0.85065081+0.j        ,  0.52573111+0.j        ],
       [ 0\.        -0.52573111j,  0\.        -0.85065081j]]))

```

* * *

### 3。点积

使用 NumPy 线性代数函数，我们可以对标量值和多维值执行点运算。它对一维向量值执行标量乘法。

对于多维数组/矩阵，它对数据值执行矩阵乘法。

**语法:**

```py
numpy.dot()

```

**举例:**

```py
import numpy as np

sc_dot = np.dot(10,2)
print("Dot Product: ", sc_dot)

vectr_x = 1 + 2j
vectr_y = 2 + 4j

vctr_dot = np.dot(vectr_x, vectr_y)
print("Dot Product: ", vctr_dot)

```

**输出:**

```py
Dot Product:  20
Dot Product:  (-6+8j)

```

* * *

### 4。用 NumPy 模块解线性方程组

有了 NumPy 线性代数函数，我们甚至可以执行计算和求解线性代数标量方程。 **numpy.linalg.solve()函数**用公式 ax=b 求解数组值。

**举例:**

```py
import numpy as np

x = np.array([[2, 4], [6, 8]])

y = np.array([2, 2])

print(("Solution of linear equations:", 
      np.linalg.solve(x, y)))

```

**输出:**

```py
('Solution of linear equations:', array([-1.,  1.]))

```

* * *

## 结论

如果你遇到任何问题，欢迎在下面评论。更多关于 Python 编程的文章，请继续关注我们。在那之前，学习愉快！！🙂