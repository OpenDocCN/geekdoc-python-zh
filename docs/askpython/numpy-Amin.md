# NumPy Amin–使用 NumPy 返回数组元素的最小值

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-Amin

大家好，欢迎来到这个关于 **Numpy amin** 的教程。在本教程中，我们将学习 **NumPy amin()** 方法，也将看到许多相同的例子。让我们开始吧！

***也读作:[Numpy . Subtract():Python 中如何用 NumPy 减去数字？](https://www.askpython.com/python-modules/numpy/numpy-subtract)***

* * *

## 什么是 NumPy amin？

NumPy 中的 amin 方法是一个返回数组元素最小值的函数。它可以是所有数组元素的最小值、沿行数组元素的最小值或沿列数组元素的最小值。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy amin 的语法

```py
numpy.amin(a, axis=None, out=None)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 输入数据。 | 需要 |
| 轴 | 沿其计算数组最小值的轴。它可以是 axis=0 或 axis=1 或 axis=None，这意味着要返回展平数组的最小值。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |

**返回:***中的最小元素 a* 。如果 *axis=None* ，则输出为标量，否则，输出为数组。

* * *

## numpy.amin()的示例

让我们进入使用 numpy.amin()函数的不同例子。

### 当数组是一维时使用 numpy.amin()

```py
import numpy as np

a = [10, 3, 25]

ans = np.amin(a)
print("a =", a)
print("Minimum of a =", ans)

```

**输出:**

```py
a = [10, 3, 25]
Minimum of a = 3

```

比较给定数组中的所有元素，10，3 和 25 的最小值是 3。因此，返回 3。

* * *

### 当数组包含负数时使用 numpy.amin()

```py
import numpy as np

a = [[-8, 6], [-5, -12]]

ans = np.amin(a)
print("a =", a)
print("Minimum of a =", ans)

```

**输出:**

```py
a = [[-8, 6], [-5, -12]]
Minimum of a = -12

```

比较数组中的所有值，-12 是这里的最小元素。

* * *

### 当数组包含 NaN 值时使用 numpy.amin()

在 Python 中，NaN 代表的不是数字。

```py
import numpy as np

a = [26, np.nan, 8, np.nan, -4]

ans = np.amin(a)
print("a =", a)
print("Minimum of a =", ans)

```

**输出:**

```py
a = [26, nan, 8, nan, -4]
Minimum of a = nan

```

如果输入包含 nan，那么 NumPy `amin()`方法总是返回 **nan** 作为输出，而不考虑输入数组中的其他元素。

* * *

### 当数组是二维数组时使用 numpy.amin()

```py
import numpy as np

a = [[16, 3], [48, 66]]

ans = np.amin(a)
print("a =", a)
print("Minimum of a =", ans)

```

**输出:**

```py
a = [[16, 3], [48, 66]]
Minimum of a = 3

```

在二维数组的情况下，当没有提到轴时，数组首先按行展平，然后计算其最小值。
在上面的例子中，展平的数组将是[16，3，48，66]并且其中的最小元素是 3，因此它由 **amin()** 方法返回。

* * *

### 使用 numpy.amin()找到给定轴上的最小值

**轴= 0**

```py
import numpy as np

a = [[16, 3], [48, 66]]
# minimum along axis=0
ans = np.amin(a, axis=0)
print("a =", a)
print("Minimum of a =", ans)

```

**输出:**

```py
a = [[16, 3], [48, 66]]
Minimum of a = [16  3]

```

这里，元素按列进行比较，它们的最小值存储在输出中。

```py
ans[0] = minimum(a[0][0], a[1][0]) = minimum(16, 48) = 16
ans[1] = minimum(a[0][1], a[1][1]) = minimum(3, 66) = 3

```

**轴= 1**

```py
import numpy as np

a = [[16, 3], [48, 66]]
# minimum along axis=1
ans = np.amin(a, axis=1)
print("a =", a)
print("Minimum of a =", ans)

```

**输出:**

```py
a = [[16, 3], [48, 66]]
Minimum of a = [ 3 48]

```

这里，元素按行进行比较，它们的最小值存储在输出中。

```py
ans[0] = minimum(a[0][0], a[0][1]) = minimum(16, 3) = 3
ans[1] = minimum(a[1][0], a[1][1]) = minimum(48, 66) = 48

```

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy amin** 方法，并使用该方法练习了不同类型的示例。如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy amin 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.amin.html)