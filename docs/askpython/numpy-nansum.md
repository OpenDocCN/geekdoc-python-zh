# NumPy nan sum–完整指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-nansum

你好，欢迎来到这个关于 **Numpy nansum** 的教程。在本教程中，我们将学习 NumPy `nansum()`方法，也将看到许多关于这个方法的例子。让我们开始吧！

***也读作:[NumPy nanprod——完全指南](https://www.askpython.com/python-modules/numpy/numpy-nanprod)***

* * *

## 什么是 NumPy nansum？

在 Python 中， **NaN** 表示**而不是数字**。如果我们有一个包含一些 NaN 值的数组，并且想要找到它的和，我们可以使用 NumPy 的`nansum()`方法。

NumPy 中的`nansum()`方法是一个函数，它返回通过将数组中的 NaN 值视为等于 0 而计算出的数组元素之和。它可以是所有数组元素的总和、沿行数组元素的总和或沿列数组元素的总和。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy nansum 的语法

```py
numpy.nansum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 需要求和的输入数组。 | 需要 |
| 轴 | 要沿其计算数组总和的轴。它可以是 axis=0，即沿列，也可以是 axis=1，即沿行，或者 axis=None，这意味着要返回整个数组的总和。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| keepdims (bool) | 如果设置为真，减少的轴将作为尺寸为 1 的尺寸留在结果中。使用此选项，结果将根据输入数组正确传播。 | 可选择的 |
| 最初的 | 总和的起始值。 | 可选择的 |
| 在哪里 | 要包含在总和中的元素。 | 可选择的 |

**返回:**
一个与 *a* 形状相同的数组，如果轴不为 None，则包含沿给定轴将 NaN 值视为 0 的 *a* 的元素之和。如果 axis=None，则返回一个标量，它是整个数组的总和。

* * *

## Numpy.nansum()的示例

现在让我们来看一些例子。

### 使用 Numpy.nansum()对整个数组求和

**一维数组**

```py
import numpy as np

a = np.array([10, np.nan, 5, 3, np.nan])
ans = np.nansum(a)
print("a =", a)
print("Sum of the array =", ans)

```

**输出:**

```py
a = [10\. nan  5\.  3\. nan]
Sum of the array = 18.0

```

这里，通过将所有 NaN 值视为零来计算数组的总和。因此，总和= 10+0+5+3+0 = 18。

**二维数组**

```py
import numpy as np

a = np.array([[10, np.nan, 5], [np.nan, 2, 6]])
ans = np.nansum(a)
print("a =", a)
print("Sum of the array =", ans)

```

**输出:**

```py
a = [[10\. nan  5.]
 [nan  2\.  6.]]
Sum of the array = 23.0

```

类似于上面的例子，sum = 10+0+5+0+2+6 = 23。

* * *

### 使用 Numpy.nansum()沿轴求和

**逐列求和**

```py
import numpy as np

a = np.array([[10, np.nan, 5], 
[np.nan, 2, 6]])
# sum along axis=0 i.e. columns 
ans = np.nansum(a, axis = 0)
print("a =", a)
print("Sum of the array =", ans)

```

**输出:**

```py
a = [[10\. nan  5.]
 [nan  2\.  6.]]
Sum of the array = [10\.  2\. 11.]

```

axis=0 指定按列计算总和。
第 0 列总和= 10+0 = 10
第 1 列总和= 0+2 =2
第 2 列总和= 5+6 = 11

**逐行求和**

```py
import numpy as np

a = np.array([[10, np.nan, 5], 
[np.nan, 2, 6]])
# sum along axis=1 i.e. rows 
ans = np.nansum(a, axis = 1)
print("a =", a)
print("Sum of the array =", ans)

```

**输出:**

```py
a = [[10\. nan  5.]
 [nan  2\.  6.]]
Sum of the array = [15\.  8.]

```

将 NaN 值视为 0，
第 0 行总和= 10+0+5 = 15
第 1 行总和= 0+2+6 = 8

* * *

### 包含无穷大的数组的和

```py
import numpy as np

# array containing +infinity
a = np.array([8, 4, np.nan, np.inf, 13])
# array containing -infinity
b = np.array([8, 4, np.nan, np.NINF, 13])
# array containing +infinity and -infinity
c = np.array([8, 4, np.nan, np.inf, np.NINF, 13])

sum_a = np.nansum(a)
sum_b = np.nansum(b)
sum_c = np.nansum(c)

print("a =", a)
print("Sum of the array a =", sum_a)
print("b =", b)
print("Sum of the array b =", sum_b)
print("c =", c)
print("Sum of the array c =", sum_c)

```

**输出:**

```py
a = [ 8\.  4\. nan inf 13.]
Sum of the array a = inf
b = [  8\.   4\.  nan -inf  13.]
Sum of the array b = -inf
c = [  8\.   4\.  nan  inf -inf  13.]
Sum of the array c = nan

```

在上面的代码中，NINF 表示-无穷大，INF 表示无穷大。请注意，如果数组包含正无穷大，则总和为正无穷大，如果数组包含负无穷大，则总和为负无穷大。如果数组同时包含正无穷大和负无穷大，则数组的和为 NaN。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy nansum** 方法，并使用相同的方法练习了不同类型的示例。如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy nansum 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.nansum.html)