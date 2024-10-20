# NumPy mod——NumPy 中模数运算符的完整指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-mod

你好，欢迎来到这个关于 **Numpy mod** 的教程。在本教程中，我们将学习 **NumPy mod()** 方法，也将看到许多关于相同的例子。让我们开始吧！

* * *

## 什么是 NumPy mod？

NumPy 中的`mod()`方法返回两个给定数组相除的元素余数。Python 中的`%`运算符也返回除法的余数，类似于`mod()`函数。

我们将在本教程接下来的章节中看到演示该函数用法的例子。

* * *

## NumPy mod 的语法

```py
numpy.mod(x1, x2, out=None)

```

| **参数** | **描述** | **必需/可选** |
| x1 (array_like) | 红利数组。 | 需要 |
| x2(类似数组) | 除数数组。 | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |

**返回:**返回除法的元素余数。如果 *x1* 和 *x2* 都是标量，那么结果也是标量值。

* * *

## 例子

现在让我们开始使用 numpy.mod 方法，这样我们可以理解输出。

### 当两个元素都是标量时

```py
import numpy as np 

dividend = 15
divisor = 7
ans = np.mod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print(dividend, "%", divisor, "=", ans)

```

**输出:**

```py
Dividend = 15 
Divisor = 7
15 % 7 = 1

```

两个元素都是标量的简单情况。7*2=14，7*3=21，所以 15 不能被 7 整除，余数在这里是 1。

* * *

### 使用 numpy.mod()对标量和数组求模

```py
import numpy as np 

dividend = [13, 8, 16]
divisor = 7
ans = np.mod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print(dividend, "%", divisor, "=", ans)

```

**输出:**

```py
Dividend = [13, 8, 16] 
Divisor = 7
[13, 8, 16] % 7 = [6 1 2]

```

在这种情况下，**被除数**数组中的所有元素都被除数一个接一个地除，并且每个除法的余数都存储在结果数组中。
输出计算如下:
13% 7 = 6
8% 7 = 1
16% 7 = 2

我们也可以如下反转元素:

```py
import numpy as np 

dividend = 7
divisor = [7, 5, 3]
ans = np.mod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print(dividend, "%", divisor, "=", ans)

```

**输出:**

```py
Dividend = 7 
Divisor = [7, 5, 3]
7 % [7, 5, 3] = [0 2 1]

```

这里，**除数**数组中的每个元素除以**被除数**即 7，余数存储在输出数组中。因此，输出被计算为:
7%7 = 0
7%5 = 2
7%3 = 1

* * *

### 当两个元素都是一维数组时的模数

```py
import numpy as np 

dividend = [30, 58, 35]
divisor = [5, 9, 4]
ans = np.mod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print(dividend, "%", divisor, "=", ans)

```

**输出:**

```py
Dividend = [30, 58, 35] 
Divisor = [5, 9, 4]
[30, 58, 35] % [5, 9, 4] = [0 4 3]

```

这里，两个数组中相同位置的元素进行除法运算，并计算余数。即被除数[0]除以除数[0]等等。这只不过是元素级的划分。
输出计算如下:

```py
dividend[0] % divisor[0] = 30%5 = 0
dividend[1] % divisor[1] = 58%9 = 4
dividend[2] % divisor[2] = 35%4 = 3

```

* * *

### 当两个元素都是二维数组时

```py
import numpy as np 

dividend = [[16, 15], [24, 23]]
divisor = [[4, 7], [10, 9]]
ans = np.mod(dividend, divisor)

print("Dividend =", dividend, "\nDivisor =", divisor)
print(dividend, "%", divisor, "=\n", ans)

```

**输出:**

```py
Dividend = [[16, 15], [24, 23]] 
Divisor = [[4, 7], [10, 9]]
[[16, 15], [24, 23]] % [[4, 7], [10, 9]] =
 [[0 1]
 [4 5]]

```

与上面 1 维数组的例子相同，这里也进行元素式除法，余数计算如下:
第 1 行:

```py
dividend[0][0] % divisor[0][0] = 16%4 = 0
dividend[0][1] % divisor[0][1] = 15%7 = 1

```

第 2 行:

```py
dividend[1][0] % divisor[1][0] = 24%10 = 4
dividend[1][1] % divisor[1][1] = 23%9 = 5

```

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy mod** 方法，并使用该方法练习了不同类型的示例。如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy mod 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.mod.html)