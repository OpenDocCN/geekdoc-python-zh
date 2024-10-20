# NumPy amax–一个轴上数组的最大值

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-amax

你好，欢迎来到这个关于 **Numpy amax** 的教程。在本教程中，我们将学习 **NumPy amax()** 方法，也将看到许多关于这个方法的例子。让我们开始吧！

***也读作: ***[NumPy fmax():数组元素的逐元素最大值](https://www.askpython.com/python-modules/numpy/numpy-fmax)******

* * *

## 什么是 NumPy amax？

[NumPy](https://www.askpython.com/python-modules/numpy) 中的`amax()`方法是一个返回数组元素最大值的函数。它可以是所有数组元素的最大值、沿行数组元素的最大值或沿列数组元素的最大值。

在 Python 中，NaN 的意思不是数字。如果在输入数据中，任何一个元素是 NaN，那么最大值也将是 NaN。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy amax 的语法

```py
numpy.amax(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 输入数据。 | 需要 |
| 轴 | 沿其计算数组最大值的轴。可以是 axis=0 或 axis=1 或 axis=None，这意味着要返回整个数组的最大值。
如果轴是整数元组，则在元组中指定的所有轴上计算最大值，而不是像以前一样在单个轴或所有轴上计算。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| keepdims (bool) | 如果设置为真，减少的轴将作为尺寸为 1 的尺寸留在结果中。使用此选项，结果将根据输入数组正确传播。 | 可选择的 |
| 最初的 | 输出元素的最小值。 | 可选择的 |
| 在哪里 | 要比较以找出最大值的元素。 | 可选择的 |

**退货:**

*a* 的最大值。如果 *a* 是标量，那么结果也是标量，否则就是数组。

* * *

## Numpy Amax 函数使用示例

让我们进入使用 **numpy.amax()** 函数的不同例子。

### 一维数组的最大值

```py
import numpy as np

arr = [3, 6, 22, 10, 84]
# using amax method to compute the maximum
ans = np.amax(arr)
print("arr =", arr)
print("Maximum value =", ans)

```

**输出:**

```py
arr = [3, 6, 22, 10, 84]
Maximum value = 84

```

* * *

### 二维数组的最大值

```py
import numpy as np

arr = [[10, 36], [4, 16]]
# using amax method to compute the maximum
ans = np.amax(arr)
print("arr =", arr)
print("Maximum value =", ans)

```

**输出:**

```py
arr = [[10, 36], [4, 16]]
Maximum value = 36

```

由于这是一个二维数组，它首先像这样按行展平:[10，36，4，16]，然后计算最大值。

* * *

### 沿轴最大值=0

```py
import numpy as np

arr = [[10, 36], [4, 16]]
# using amax method to compute the maximum
ans = np.amax(arr, axis=0)
print("arr =", arr)
print("Maximum value =", ans)

```

**输出:**

```py
arr = [[10, 36], [4, 16]]
Maximum value = [10 36]

```

这里，最大值沿列计算如下:

```py
ans[0] = max(arr[0][0], arr[1][0]) = max(10, 4) = 10
ans[1] = max(arr[0][1], arr[1][1]) = max(36, 16) = 36

```

* * *

### 沿轴最大值=1

```py
import numpy as np

arr = [[10, 36], [4, 16]]
# using amax method to compute the maximum
ans = np.amax(arr, axis=1)
print("arr =", arr)
print("Maximum value =", ans)

```

**输出:**

```py
arr = [[10, 36], [4, 16]]
Maximum value = [36 16]

```

在这种情况下，最大值沿行计算如下:

```py
ans[0] = max(arr[0][0], arr[0][1]) = max(10, 36) = 36
ans[1] = max(arr[1][0], arr[1][1]) = max(4, 16) = 16

```

* * *

### 包含 NaN 的数组的最大值

```py
import numpy as np

arr = [3, 6, np.nan, 22, np.nan, 10, 84]
# using amax method to compute the maximum
ans = np.amax(arr)
print("arr =", arr)
print("Maximum value =", ans)

```

**输出:**

```py
arr = [3, 6, nan, 22, nan, 10, 84]
Maximum value = nan

```

如本教程前面所述，如果一个数组包含 NaN，那么它的最大值也是 NaN，如上面的例子所示。

* * *

### 给定初始值的数组的最大值

```py
import numpy as np

arr = [3, 6, 22, 10, 84]
# using amax method to compute the maximum
ans = np.amax(arr, initial=100)
print("arr =", arr)
print("Maximum value =", ans)

```

**输出:**

```py
arr = [3, 6, 22, 10, 84]
Maximum value = 100

```

这里，我们提到了一个初始值为 100 的*。将该值与数组中的所有元素进行比较，以找到最大值。
在这里，因为 100 是所有值中最高的值，所以它被返回。*

* * *

### 仅使用选定元素的数组的最大值

为了找到数组中某些选择值的最大值，我们可以将 *where* 参数传递给 **numpy.amax()** 函数。

```py
import numpy as np

arr = [3, 6, 22, 10, 84]
# using amax method to compute the maximum
ans = np.amax(arr, where=[False, False, True, True, False], initial=-1)
print("arr =", arr)
print("Maximum value =", ans)

```

**输出:**

```py
arr = [3, 6, 22, 10, 84]
Maximum value = 22

```

这里在*里*列表只有指标 2 和 3 是**真**，其余都是**假**。这意味着 amax()方法必须只找到 *arr* 中索引 2 和 3 处元素的最大值，并忽略其余元素。
因此，返回的答案是 max(22，10) = 10。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy amax** 方法，并使用相同的方法练习了不同类型的示例。
如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy amax 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.amax.html#numpy.amax)