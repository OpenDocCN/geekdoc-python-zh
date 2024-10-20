# NumPy 闵楠–忽略任何 NaNs 的沿轴数组的最小值

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-nanmin>

你好，欢迎来到这个关于 **Numpy 闵楠**的教程。在本教程中，我们将学习 NumPy `nanmin()`方法，也将看到许多关于这个方法的例子。让我们开始吧！

***也读作:[NumPy nanmax——忽略任何 NaNs 的沿轴数组的最大值](https://www.askpython.com/python-modules/numpy/numpy-nanmax)***

* * *

## 什么是 NumPy 闵楠？

在 Python 中， **NaN** 表示**而不是数字**。如果我们有一个包含一些 NaN 值的数组，并且想要找到其中的最小值，我们可以使用 NumPy 的`nanmin()`方法。

NumPy 中的`nanmin()`方法是一个函数，它返回通过忽略数组中的 NaN 值而计算出的数组元素的最小值。它可以是所有数组元素的最小值、沿行数组元素的最小值或沿列数组元素的最小值。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy 闵楠的语法

让我们来看看`nanmin()`函数的语法。

```py
numpy.nanmin(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 输入数据。 | 需要 |
| 轴 | 沿其计算数组最小值的轴。可以是 axis=0 或 axis=1 或 axis=None，这意味着要返回整个数组的最小值。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| keepdims (bool) | 如果设置为真，减少的轴将作为尺寸为 1 的尺寸留在结果中。使用此选项，结果将根据输入数组正确传播。 | 可选择的 |
| 最初的 | 输出元素的最大值。 | 可选择的 |
| 在哪里 | 要比较以找出最小值的元素。 | 可选择的 |

NumPy nanmin parameters

**返回:**
一个数组，包含沿指定轴的数组的最小值，忽略所有的 NaNs。

* * *

## NumPy 闵楠的例子

让我们进入使用 **numpy.nanmin()** 函数的不同例子。

### 一维数组的 NumPy 闵楠

```py
import numpy as np

arr = [np.nan, 54, 1, 3, 44]
# using the nanmin function to calculate the maximum
ans = np.nanmin(arr)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [nan, 54, 1, 3, 44]
Result = 1.0

```

忽略 NaN 值，54、1、3 和 44 中的最小值是 1，因此返回该值。

* * *

### 二维数组的闵楠数

```py
import numpy as np

arr = [[30, -9], [8, np.nan]]
# using the nanmin function to calculate the maximum
ans = np.nanmin(arr)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [[30, -9], [8, nan]]
Result = -9.0

```

与前面的例子类似，30、-9 和 8 的最小值是 8。

* * *

### 沿着阵列轴的闵楠数

**轴= 0**

```py
import numpy as np

arr = [[16, 4], [np.nan, 1]]
# using the nanmin function to calculate the maximum
ans = np.nanmin(arr, axis=0)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [[16, 4], [nan, 1]]
Result = [16\.  1.]

```

这里，比较特定列的每一行中的值，以找到最小元素。

```py
ans[0] = min(arr[0][0], arr[1][0]) = min(16, np.nan) = 16 (ignoring NaN)
ans[1] = min(arr[0][1], arr[1][1]) = min(4, 1) = 1

```

**轴= 1**

```py
import numpy as np

arr = [[16, 4], [np.nan, 1]]
# using the nanmin function to calculate the maximum
ans = np.nanmin(arr, axis=1)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [[16, 4], [nan, 1]]
Result = [4\. 1.]

```

当*轴=1* 时，每一行中的元素在所有列中进行比较，以找到最小值。

```py
ans[0] = min(arr[0][0], arr[0][1]) = min(16, 4) = 4
ans[1] = min(arr[1][0], arr[1][1]) = min(np.nan, 1) = 1 (ignoring NaN)

```

* * *

### 包含无穷大的数组的 NumPy 闵楠

现在让我们看看`numpy.nanmin()`方法如何处理数组中的无穷大和 nan。

```py
import numpy as np

# array containing +infinity
a = np.array([16, 3, np.nan, 7, np.inf])
# array containing -infinity
b = np.array([16, 3, np.nan, 7, np.NINF])
# array containing +infinity and -infinity
c = np.array([16, 3, np.nan, np.NINF, 7, np.inf])

min_a = np.nanmin(a)
min_b = np.nanmin(b)
min_c = np.nanmin(c)

print("a =", a)
print("Minimum of the array a =", min_a)
print("\nb =", b)
print("Minimum of the array b =", min_b)
print("\nc =", c)
print("Minimum of the array c =", min_c)

```

**输出:**

```py
a = [16\.  3\. nan  7\. inf]
Minimum of the array a = 3.0

b = [ 16\.   3\.  nan   7\. -inf]
Minimum of the array b = -inf

c = [ 16\.   3\.  nan -inf   7\.  inf]
Minimum of the array c = -inf

```

在上面的代码中， **NINF** 表示**-无穷大**， **inf** 表示**无穷大**。请注意，

*   如果数组包含**正无穷大**，那么最小值是忽略 NaNs 的整个数组的最小值。
*   如果数组包含**负无穷大**，那么最小值就是**负无穷大**。
*   如果数组包含**正负无穷大**，那么数组的最小值是 **-inf** ，即**负无穷大**。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy 闵楠**方法，并使用相同的方法练习了不同类型的例子。
如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy 闵楠官方文档](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html)