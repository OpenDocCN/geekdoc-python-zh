# NumPy mat mul–两个数组的矩阵乘积

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-matmul

你好，欢迎来到这个关于 Numpy matmul 的教程。在本教程中，我们将学习 **NumPy matmul()** 方法，也将看到许多关于这个方法的例子。让我们开始吧！

* * *

## 什么是 NumPy matmul？

NumPy 中的`matmul()`方法返回两个数组的矩阵乘积。这里，输入参数只能是数组，不允许有标量值。输入 be 可以是 1 维数组、2 维数组或两者的组合，或者也可以是 n 维数组。

我们将在本教程接下来的章节中看到这些例子。

* * *

## NumPy matmul 的语法

让我们来看看`matmul`函数的语法。

```py
numpy.matmul(x1, x2, out=None)

```

| **参数** | **描述** | **必需/可选** |
| x1 | 输入数组 1。 | 需要 |
| x2 | 输入数组 2。 | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |

如果 *x1* 是一个*n×m*矩阵， *x2* 是*m×l*矩阵，那么相乘后得到的矩阵将是一个*n×l*矩阵。

**返回:**
*x1*nad*x2*的矩阵乘积。如果 *x1* 和 *x2* 都是一维数组，那么结果将是一个标量值。

**如果 *x1* 的最后一个维度与 *x2* 的倒数第二个维度不匹配，或者如果一个标量值作为参数传递，则引发:**
。

* * *

## 使用 NumPy matmul 的示例

现在让我们看几个例子来更好地理解这个函数。

### 当两个输入都是一维数组时使用 NumPy matmul

```py
import numpy as np

a = [1, 5, 3]
b = [10, 2, 4]
# using matmul method to compute the matrix product
ans = np.matmul(a, b)
print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = [1, 5, 3] 
b = [10, 2, 4]
Result = 32

```

这里，因为两个输入参数都是一维数组，所以它们的矩阵乘法会产生一个标量值，计算如下

```py
ans = 1*10 + 5*2 + 3*4 = 10 + 10 + 12 = 32 

```

* * *

### 当两个输入都是二维数组时

```py
import numpy as np

a = [[2, 6], [8, 4]]
b = [[3, 1], [5, 10]]
# using matmul method to compute the matrix product
ans = np.matmul(a, b)
print("a =", a, "\nb =", b)
print("Result =\n", ans)

```

**输出:**

```py
a = [[2, 6], [8, 4]] 
b = [[3, 1], [5, 10]]
Result =
 [[36 62]
 [44 48]]

```

由于两个输入都是 2×2 矩阵，因此结果也是 2×2 矩阵。矩阵乘法计算如下

```py
ans[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] = 2*3 + 6*5 = 6 + 30 = 36
ans[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] = 2*1 + 6*10 = 2 + 60 = 62
ans[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] = 8*3 + 4*5 = 24 + 20 = 44
ans[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] = 8*1 + 4*10 = 8 + 40 = 48

```

* * *

### 当一个输入是一维数组而另一个是二维数组时使用 NumPy matmul

```py
import numpy as np

a = [10, 20]
b = [[8, 9], [3, 1]]
# using matmul method to compute the matrix product
ans = np.matmul(a, b)
print("a =", a, "\nb =", b)
print("Matrix product of a and b =", ans)

```

**输出:**

```py
a = [10, 20] 
b = [[8, 9], [3, 1]]
Matrix product of a and b = [140 110]

```

矩阵 *a* 的形状是 1×2，而 *b* 的形状是 2×2，因此得到的矩阵的形状是 1×2。矩阵乘积的计算如下:

```py
ans[0][0] = a[0][0]*b[0][0] + a[0][1]*b[0][1] = 10*8 + 20*3 = 80 + 60 = 140
ans[0][1] = a[0][0]*b[1][0] + a[0][1]*b[1][1] = 10*9 + 20*1 = 90 + 20 = 110 

```

我们也可以颠倒 matmul 函数中矩阵的顺序，如下所示:

```py
import numpy as np

a = [10, 20]
b = [[8, 9], [3, 1]]
# using matmul method to compute the matrix product
ans = np.matmul(b, a)
print("a =", a, "\nb =", b)
print("Matrix product of b and a =", ans)

```

**输出:**

```py
a = [10, 20] 
b = [[8, 9], [3, 1]]

```

这里，输出计算如下:

```py
ans[0][0] = b[0][0]*a[0][0] + b[1][0]*a[0][1] = 8*10 + 9*20 = 80 + 180 = 260
ans[0][1] = b[0][1]*a[0][0] + b[1][1]*a[0][1] = 3*10 + 1*20 = 30 + 20 = 50 

```

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy matmul** 方法，并使用相同的方法练习了不同类型的示例。如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy matmul 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)