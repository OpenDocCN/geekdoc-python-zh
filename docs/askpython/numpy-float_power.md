# NumPy float_power

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-float_power>

你好，欢迎来到这个关于 **Numpy float_power** 的教程。在本教程中，我们将学习 **NumPy float_power()** 方法，也将看到许多关于这个方法的例子。让我们开始吧！

***也读作: [NumPy 幂——将另一个数提升到](https://www.askpython.com/python-modules/numpy/numpy-power)*** 的幂

* * *

## 什么是 NumPy float_power？

NumPy 中的`float_power()`方法是一个函数，它返回一个数组，该数组是通过对一个数组中的元素进行与第二个数组中的值相对应的幂来计算的。

如果 *x1* 和 *x2* 是两个数组，那么`float_power(x1, x2)`按元素计算输出，即通过将 *x1* 中的每个值提高到 *x2* 中相应位置的值。正如函数本身的名字所暗示的，它的默认返回类型是 float。我们将在本教程的下一节看到这个函数的例子。

* * *

## NumPy float_power 语法

```py
numpy.float_power(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

```

| **参数** | **描述** | **必需/可选** |
| x1 (array_like) | 基本数组。 | 需要 |
| x2(类似数组) | 幂/指数数组。 | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| 在哪里 | 接受一个类似数组的对象。在为真的位置，`out`数组将被设置为`ufunc`结果。在其他地方，`out`数组将保持其原始值。 | 可选择的 |

**返回:**
一个数组，包含 *x* 1 提升到 *x2* 的结果，按元素排序。如果 *x1* 和 *x2* 是标量，那么结果也是标量。

* * *

## Numpy float_power()的示例

现在让我们进入例子，理解`float_power`方法实际上是如何工作的。

### 当两个输入都是标量时

```py
import numpy as np

a = 5
b = 3
# using the float_power method 
ans = np.float_power(a, b)

print("a = ", a, "\nb =", b)
print("Result = ", ans)

```

**输出:**

```py
a =  5 
b = 3
Result =  125.0

```

一个简单的例子，结果计算如下

```py
5^3 = 125

```

* * *

### 当一个输入是标量而另一个是一维数组时

```py
import numpy as np

a = [0, -2, 4, -6, 8]
b = 3
# using the float_power method 
ans = np.float_power(a, b)

print("a = ", a, "\nb =", b)
print("Result = ", ans)

```

**输出:**

```py
a =  [0, -2, 4, -6, 8] 
b = 3
Result =  [   0\.   -8\.   64\. -216\.  512.]

```

这里，数组 *a* 中的每个元素被提升到幂 *b* 并且输出被计算为

```py
ans[0] = a[0] ^ b = 0 ^ 3 = 0
ans[1] = a[1] ^ b = -2 ^ 3 = -8
ans[2] = a[2] ^ b = 4 ^ 3 = 64
ans[3] = a[3] ^ b = -6 ^ 3 = -216
ans[4] = a[4] ^ b = 8 ^ 3 = 512

```

从输出中，我们可以看到该函数也处理负值。

* * *

### 当两个输入阵列都是一维时

```py
import numpy as np

a = [3, 1, 4, 2.5]
b = [0, 2, 2.7, 4]
# using the float_power method 
ans = np.float_power(a, b)

print("a = ", a, "\nb =", b)
print("Result = ", ans)

```

**输出:**

```py
a =  [3, 1, 4, 2.5] 
b = [0, 2, 2.7, 4]
Result =  [ 1\.          1\.         42.22425314 39.0625    ]

```

这里， *a* 中的每个元素被提升到 *b* 中相应元素的幂，输出被计算为:

```py
ans[0] = a[0] ^ b[0] = 3 ^ 0 = 1
ans[1] = a[1] ^ b[1] = 1 ^ 2 = 1
ans[2] = a[2] ^ b[2] = 4 ^ 2.7 = 42.22425314
ans[3] = a[3] ^ b[3] = 2.5 ^ 4 = 39.0625

```

注意偶数浮点数是由`float_power()`方法处理的。

* * *

### 当两个输入阵列都是二维时

```py
import numpy as np

a = [[1, 2], [6, 3]]
b = [[2, 5], [2, 3]]
# using the float_power method 
ans = np.float_power(a, b)

print("a = ", a, "\nb =", b)
print("Result = \n", ans)

```

**输出:**

```py
a =  [[1, 2], [6, 3]] 
b = [[2, 5], [2, 3]]
Result = 
 [[ 1\. 32.]
 [36\. 27.]]

```

类似于上面的例子，

```py
ans[0][0] = a[0][0] ^ b[0][0] = 1 ^ 2 = 1
ans[0][1] = a[0][1] ^ b[0][1] = 2 ^ 5 = 32

ans[1][0] = a[1][0] ^ b[1][0] = 6 ^ 2 = 36
ans[1][1] = a[1][1] ^ b[1][1] = 3 ^ 3 = 27

```

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy float_power** 方法，并使用相同的方法练习了不同类型的示例。

如果你想了解更多关于 NumPy 的知识，请随意浏览我们的 NumPy 教程。

* * *

## 参考

*   [NumPy float_power 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html)