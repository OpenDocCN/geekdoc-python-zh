# NumPy div mod–返回元素的商和余数

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-divmod

你好，欢迎来到这个关于 Numpy divmod 的教程。在本教程中，我们将学习 **NumPy divmod()** 方法，也将看到许多关于相同的例子。让我们开始吧！

***也读:[NumPy mod——NumPy](https://www.askpython.com/python-modules/numpy/numpy-mod)*中模数运算符的完全指南**

* * *

## 什么是 NumPy divmod？

NumPy 中的`divmod()`方法返回两个给定数组的**元素商和除法余数**。Python 中的`//`返回商的底，而`%`运算符返回除法的余数，类似于`mod()`函数。

在本教程接下来的章节中，我们将看到演示这个函数用法的例子。

* * *

## NumPy divmod 的语法

```py
numpy.divmod(x1, x2, out=None)

```

| **参数** | **描述** | **必需/可选** |
| x1 (array_like) | 红利数组。 | 需要 |
| x2(类似数组) | 除数数组。 | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |

**返回:**
返回除法运算的元素商和余数。如果 *x1* 和 *x2* 都是标量，那么结果也是标量值。

**关于 divmod()函数的一些常见观察:**

*   如果第一个参数，即 *x1* 为 0，则该函数返回(0，0)。
*   如果第二个参数，即 *x2* 为 0，则该函数返回一个*‘零除法误差’*。
*   如果 *x1* 是浮点值，函数返回(q，x1%x2)，其中 *q* 是商的整数部分。

* * *

## numpy.divmod()的示例

现在让我们开始使用 **numpy.divmod** 方法，这样我们可以理解输出。

### 当两个元素都是标量时，使用 numpy.divmod()

```py
import numpy as np 

dividend = 23
divisor = 6
ans = np.divmod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print("Result =", ans)

```

**输出:**

```py
Dividend = 23 
Divisor = 6
Result = (3, 5)

```

这里两个元素都是标量。6*3=18，6*4=24，所以 23 不能被 6 整除。当 23 除以 6 时，商是 3，余数是 23-18=5，返回为(3，5)。

* * *

### 当一个元素是标量而另一个是数组时，使用 numpy.divmod()

```py
import numpy as np 

dividend = [30, 19, 8]
divisor = 6
ans = np.divmod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print("Result =", ans)

```

**输出:**

```py
Dividend = [30, 19, 8] 
Divisor = 6
Result = (array([5, 3, 1], dtype=int32), array([0, 1, 2], dtype=int32))

```

在这种情况下，**被除数**数组中的所有元素都被**除数**逐个除，并且这些除的商和余数都存储在各自的结果数组中。
输出中的**第一个数组**是**商数组**，第**第二个**是**余数数组**。

输出计算如下:

**商数组:**

```py
30//6 = 5
19//6 = 3
8//6 = 1

```

**余数数组:**

```py
30%6 = 0
19%6 = 1
8%6 = 2

```

* * *

### 当两个元素都是一维数组时，使用 numpy.divmod()

```py
import numpy as np 

dividend = [72, 60, 30]
divisor = [3, 15, 24]
ans = np.divmod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print("Result =", ans)

```

**输出:**

```py
Dividend = [72, 60, 30] 
Divisor = [3, 15, 24]
Result = (array([24,  4,  1], dtype=int32), array([0, 0, 6], dtype=int32))

```

这里，两个数组中相同位置的元素进行除法运算，并计算商和余数。即被除数[0]除以除数[0]等等。这只不过是元素级的划分。

**商数组:**

```py
dividend[0] // divisor[0] = 72//3 = 24
dividend[1] // divisor[1] = 60//15 = 4
dividend[2] // divisor[2] = 30//24 = 1

```

**余数数组:**

```py
dividend[0] % divisor[0] = 72%3 = 0
dividend[1] % divisor[1] = 60%15 = 0
dividend[2] % divisor[2] = 30%24 = 6

```

* * *

### 当两个元素都是二维数组时，使用 numpy.divmod()

```py
import numpy as np 

dividend = [[18, 35], [10, 7]]
divisor = [[5, 7], [10, 4]]
ans = np.divmod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print("Result =\n", ans)

```

**输出:**

```py
Dividend = [[18, 35], [10, 7]] 
Divisor = [[5, 7], [10, 4]]
Result =
 (array([[3, 5],
       [1, 1]], dtype=int32), array([[3, 0],
       [0, 3]], dtype=int32))

```

与上面的一维数组示例相同，这里也进行元素式除法，商和余数计算如下:

**商数组:**

```py
dividend[0][0] // divisor[0][0] = 18//5 = 3
dividend[0][1] // divisor[0][1] = 35//7 = 5
dividend[1][0] // divisor[1][0] = 10//10 = 1
dividend[1][1] // divisor[1][1] = 7//4 = 1

```

导致[[3，5]，[1，1]]。

**余数数组:**

```py
dividend[0][0] // divisor[0][0] = 18%5 = 3
dividend[0][1] // divisor[0][1] = 35%7 = 0
dividend[1][0] // divisor[1][0] = 10%10 = 0
dividend[1][1] // divisor[1][1] = 7%4 = 3

```

产生[[3，0]，[0，3]]。

* * *

## 摘要

仅此而已！在本教程中，我们学习了 **Numpy divmod** 方法，并使用该方法练习了不同类型的示例。如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy divmod 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html)