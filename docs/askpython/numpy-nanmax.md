# NumPy nan max–忽略任何 nan 的沿轴数组的最大值

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-nanmax

你好，欢迎来到这个关于 **Numpy nanmax** 的教程。在本教程中，我们将学习 NumPy `nanmax()`方法，也将看到许多关于这个方法的例子。让我们开始吧！

***也读作:[NumPy amax——沿轴最大值数组的最大值](https://www.askpython.com/python-modules/numpy/numpy-amax)***

* * *

## 什么是 NumPy nanmax？

在 Python 中， **NaN** 表示**而不是数字**。如果我们有一个包含一些 NaN 值的数组，并且想要找到其中的最大值，我们可以使用 NumPy 的`nanmax()`方法。

NumPy 中的`nanmax()`方法是一个函数，它返回通过忽略数组中的 NaN 值计算的数组元素的最大值。它可以是所有数组元素的最大值、沿行数组元素的最大值或沿列数组元素的最大值。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy nanmax 的语法

让我们来看看`nanmax()`函数的语法。

```py
numpy.nanmax(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 输入数据。 | 需要 |
| 轴 | 沿其计算数组最大值的轴。可以是 axis=0 或 axis=1 或 axis=None，这意味着要返回整个数组的最大值。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| keepdims (bool) | 如果设置为真，减少的轴将作为尺寸为 1 的尺寸留在结果中。使用此选项，结果将根据输入数组正确传播。 | 可选择的 |
| 最初的 | 输出元素的最小值。 | 可选择的 |
| 在哪里 | 要比较以找出最大值的元素。 | 可选择的 |

**返回:**
一个数组，包含数组沿指定轴的最大值，忽略所有的 nan。

* * *

## NumPy nanmax 的示例

让我们进入使用 **numpy.nanmax()** 函数的不同例子。

### 一维数组的 Nanmax

```py
import numpy as np

arr = [5, 32, 10, np.nan, 4]
# using the nanmax function to calculate the maximum
ans = np.nanmax(arr)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [5, 32, 10, nan, 4]
Result = 32.0

```

忽略 NaN 值，5、32、10 和 4 中的最大值是 32，因此返回该值。

* * *

### 二维数组的 Nanmax

```py
import numpy as np

arr = [[-12, 3], [np.nan, 36]]
# using the nanmax function to calculate the maximum
ans = np.nanmax(arr)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [[-12, 3], [nan, 36]]
Result = 36.0

```

与前面的示例类似，12、3 和 36 的最大值是 36。

* * *

### 沿阵列轴的 Nanmax

#### 轴= 0

```py
import numpy as np

arr = [[5, 8], [np.nan, 36]]
# using the nanmax function to calculate the maximum
ans = np.nanmax(arr, axis=0)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [[5, 8], [nan, 36]]
Result = [ 5\. 36.]

```

这里，比较特定列的每一行中的值，以找到最大元素。

```py
ans[0] = max(arr[0][0], arr[1][0]) = max(5, np.nan) = 5 (ignoring NaN)
ans[1] = max(arr[0][1], arr[1][1]) = max(8, 36) = 36

```

#### 轴= 1

```py
import numpy as np

arr = [[5, 8], [np.nan, 36]]
# using the nanmax function to calculate the maximum
ans = np.nanmax(arr, axis=1)

print("arr =", arr)
print("Result =", ans)

```

**输出:**

```py
arr = [[5, 8], [nan, 36]]
Result = [ 8\. 36.]

```

当*轴=1* 时，每一行中的元素在所有列中进行比较，以找到最大值。

```py
ans[0] = max(arr[0][0], arr[0][1]) = max(5, 8) = 8
ans[1] = max(arr[1][0], arr[1][1]) = max(np.nan, 36) = 36 (ignoring NaN)

```

* * *

### 包含无穷大的数组的 NumPy nanmax

现在让我们看看`numpy.nanmax()`方法如何处理数组中的无穷大和 nan。

```py
import numpy as np

# array containing +infinity
a = np.array([25, np.nan, 36, np.inf, 8])
# array containing -infinity
b = np.array([25, np.nan, 36, np.NINF, 8])
# array containing +infinity and -infinity
c = np.array([25, np.nan, 36, np.inf, np.NINF, 8])

max_a = np.nanmax(a)
max_b = np.nanmax(b)
max_c = np.nanmax(c)

print("a =", a)
print("Maximum of the array a =", max_a)
print("\nb =", b)
print("Maximum of the array b =", max_b)
print("\nc =", c)
print("Maximum of the array c =", max_c)

```

**输出:**

```py
a = [25\. nan 36\. inf  8.]
Maximum of the array a = inf

b = [ 25\.  nan  36\. -inf   8.]
Maximum of the array b = 36.0

c = [ 25\.  nan  36\.  inf -inf   8.]
Maximum of the array c = inf

```

在上面的代码中， **NINF** 表示**-无穷大**， **inf** 表示**无穷大**。请注意，

*   如果数组包含**正无穷大**，那么最大值是**正无穷大**。
*   如果数组包含负无穷大的**，那么最大值就是所有元素的**最大值，忽略 NaNs** 。**
*   如果数组包含**正无穷大和负无穷大**，那么数组的最大值是 **inf** ，即**正无穷大**。

* * *

## 摘要

仅此而已！在本教程中，我们学习了 **Numpy nanmax** 方法，并使用相同的方法练习了不同类型的示例。
如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy nanmax 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html)