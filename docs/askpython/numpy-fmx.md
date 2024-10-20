# NumPy fmax–数组元素的最大元素数

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-fmx

你好，欢迎来到这个关于 **Numpy fmax** 的教程。在本教程中，我们将学习 **NumPy fmax()** 方法，也将看到许多关于这个方法的例子。让我们开始吧！

***亦读:[【NumPy ones _ like——完全指南](https://www.askpython.com/python-modules/numpy/numpy-ones_like)***

* * *

## 什么是 NumPy fmax？

`fmax()`是 [NumPy](https://www.askpython.com/python-modules/numpy) 中的一个函数，它比较两个数组并返回一个包含这两个数组的元素最大值的数组。

* * *

## NumPy fmax 的语法

让我们来看看`fmax()`函数的语法。

```py
numpy.fmax(x1, x2, out=None)

```

| **参数** | **描述** | **必需/可选** |
| x1 | 输入数组 1。 | 需要 |
| x2 | 输入数组 2。 | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |

**返回:**
包含 *x1* 和 *x2* 的元素最大值的新数组。

*   如果 *x1* 和 *x2* 都是标量，那么输出也是标量。
*   如果 *x1* 或 *x2* 中的任何一个包含 NaN 值，则该逐元素比较的输出是非 NaN 值。
*   如果比较中的两个元素都是 NaN，则返回第一个 NaN。

* * *

## 使用 NumPy fmax 的示例

现在让我们看几个例子来更好地理解`fmax()`函数。

### 当两个输入都是标量时使用 NumPy fmax

```py
import numpy as np

a = 15
b = 8
# using fmax function to calculate the element-wise maximum
ans = np.fmax(a, b)
print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = 15 
b = 8
Result = 15

```

既然 15>8，那么答案就是 15。

* * *

### 一维数组的元素最大值

```py
import numpy as np

a = [2, 36, 1, 5, 10]
b = [6, 3 ,48, 2, 18]
# using fmax function to calculate the element-wise maximum
ans = np.fmax(a, b)
print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = [2, 36, 1, 5, 10] 
b = [6, 3, 48, 2, 18]
Result = [ 6 36 48  5 18]

```

生成的数组计算如下:

```py
ans[0]  = max(a[0], b[0]) = max(2, 6) = 6
ans[1]  = max(a[1], b[1]) = max(36, 3) = 36
ans[2]  = max(a[2], b[2]) = max(1, 48) = 48
ans[3]  = max(a[3], b[3]) = max(5, 2) = 5
ans[4]  = max(a[4], b[4]) = max(10, 18) = 18

```

* * *

### 二维数组的元素最大值

```py
import numpy as np

a = [[6, -8, 4], [2, 21, 16]]
b = [[-5, -12, 1], [0, 10, 27]]
# using fmax function to calculate the element-wise maximum
ans = np.fmax(a, b)
print("a =", a, "\nb =", b)
print("Result =\n", ans)

```

**输出:**

```py
a = [[6, -8, 4], [2, 21, 16]] 
b = [[-5, -12, 1], [0, 10, 27]]
Result =
 [[ 6 -8  4]
 [ 2 21 27]]

```

这里，两个输入数组都是 2×3 数组，因此结果数组也是 2×3 数组，计算如下:

```py
ans[0][0] = max(a[0][0], b[0][0]) = max(6, -5) = 6
ans[0][1] = max(a[0][1], b[0][1]) = max(-8, -12) = -8
ans[0][2] = max(a[0][2], b[0][2]) = max(4, 1) = 4

ans[1][0] = max(a[1][0], b[1][0]) = max(2, 0) = 2
ans[1][1] = max(a[1][1], b[1][1]) = max(21, 10) = 21
ans[1][2] = max(a[1][2], b[1][2]) = max(16, 27) = 27

```

* * *

### 包含 nan 的数组的元素最大值

```py
import numpy as np

a = [8, np.nan, 5, 3]
b = [0, np.nan, np.nan, -6]
# using fmax function to calculate the element-wise maximum
ans = np.fmax(a, b)
print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = [8, nan, 5, 3] 
b = [0, nan, nan, -6]
Result = [ 8\. nan  5\.  3.]

```

这里，

```py
ans[0] = max(a[0], b[0]) = max(8, 0) = 8

```

现在，a[1]和 b[1]都是 NaN，所以这些中的最大值也作为 NaN 返回。

```py
ans[1] = NaN

```

a[2] = 5，b[2] = NaN，因此最大值是非 NaN 值，即 5。

```py
ans[2] = 5

ans[3] = max(a[3], b[3]) = max(3, -6) = 3

```

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy fmax** 方法，并使用相同的方法练习了不同类型的示例。
如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy fmax 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.fmax.html)