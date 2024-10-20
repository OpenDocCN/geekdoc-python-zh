# NumPy fmin–数组元素的最小元素数

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-fmin>

大家好，欢迎来到这个关于 **Numpy fmin** 的教程。在本教程中，我们将学习 **NumPy fmin()** 方法，也将看到许多关于这个方法的例子。让我们开始吧！

***也读作:[NumPy fmax–数组元素的逐元素最大值](https://www.askpython.com/python-modules/numpy/numpy-fmax)***

* * *

## 什么是 NumPy fmin？

`fmin()`是 [NumPy](https://www.askpython.com/python-modules/numpy) 中的一个函数，它比较两个数组并返回一个包含这两个数组的元素最小值的数组。

* * *

## NumPy fmin 的语法

让我们来看看`fmin()`函数的语法。

```py
numpy.fmin(x1, x2, out=None)

```

| **参数** | **描述** | **必需/可选** |
| x1 | 输入数组 1。 | 需要 |
| x2 | 输入数组 2。 | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |

**返回:**
包含 *x1* 和 *x2* 的元素最大值的新数组。

*   如果 *x1* 和 *x2* 都是标量，那么输出也是标量。
*   如果 *x1* 或 *x2* 中的任何一个包含 NaN 值，则该逐元素比较的输出是非 NaN 值。
*   如果比较中的两个元素都是 NaN，则 NaN 作为最小元素返回。

* * *

## 例子

现在让我们看几个例子来更好地理解`fmin()`函数。

### 当两个输入都是标量时

```py
import numpy as np 

a = 2
b = 6
# using fmin function to calculate the element-wise minimum
ans = np.fmin(a, b)
print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = 2 
b = 6
Result = 2

```

因为 2<6，所以 2 是这里的最小元素。

* * *

### 一维数组的逐元素最小值

```py
import numpy as np 

a = [5, 3, -5, 8, -2]
b = [1, 8, -2, 12, -13]
# using fmin function to calculate the element-wise minimum
ans = np.fmin(a, b)
print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = [5, 3, -5, 8, -2] 
b = [1, 8, -2, 12, -13]
Result = [  1   3  -5   8 -13]

```

生成的数组计算如下:

```py
ans[0]  = min(a[0], b[0]) = min(5, 1) = 1
ans[1]  = min(a[1], b[1]) = min(3, 8) = 3
ans[2]  = min(a[2], b[2]) = min(-5, -2) = -5
ans[3]  = min(a[3], b[3]) = min(8, 12) = 8
ans[4]  = min(a[4], b[4]) = min(-2, -13) = -13

```

* * *

### 二维数组的逐元素最小值

```py
import numpy as np 

a = [[13, 8], [10, 7]]
b = [[5, 15], [30, 4]]
# using fmin function to calculate the element-wise minimum
ans = np.fmin(a, b)
print("a =", a, "\nb =", b)
print("Result =\n", ans)

```

**输出:**

```py
a = [[13, 8], [10, 7]] 
b = [[5, 15], [30, 4]]
Result =
 [[ 5  8]
 [10  4]]

```

这里，两个输入数组都是 2×2 数组，因此得到的数组也是 2×2 数组，计算如下:

```py
ans[0][0] = min(a[0][0], b[0][0]) = min(13, 5) = 5
ans[0][1] = min(a[0][1], b[0][1]) = min(8, 15) = 8

ans[1][0] = min(a[1][0], b[1][0]) = min(10, 30) = 10
ans[1][1] = min(a[1][1], b[1][1]) = min(7, 4) = 4

```

* * *

### 包含 nan 的数组的逐元素最小值

现在让我们看看`numpy.fmin()`方法是如何处理 nan 的。

```py
import numpy as np 

a = [4, 3, 10, np.nan, np.nan]
b = [2, np.nan, 5, 8, np.nan]
# using fmin function to calculate the element-wise minimum
ans = np.fmin(a, b)
print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = [4, 3, 10, nan, nan] 
b = [2, nan, 5, 8, nan]
Result = [ 2\.  3\.  5\.  8\. nan]

```

这里，

```py
ans[0]  = min(a[0], b[0]) = min(4, 2) = 2
ans[1]  = min(a[1], b[1]) = min(3, nan) = 3
ans[2]  = min(a[2], b[2]) = min(10, 5) = 5
ans[3]  = min(a[3], b[3]) = min(nan, 8) = 8
ans[4]  = min(a[4], b[4]) = min(nan, nan) = nan

```

在上面的数组中，索引 1 和 3 处的元素之一是 nan，因此最小值是非 NaN 值。此外，两个输入数组中索引 4 处的元素都是 NaN，因此得出的最小值也是 NaN，如本教程前面所述。

* * *

## 摘要

仅此而已！在本教程中，我们学习了 **Numpy fmin** 方法，并使用相同的方法练习了不同类型的示例。
如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy fmin 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.fmin.html)