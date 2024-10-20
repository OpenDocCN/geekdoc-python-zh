# NumPy Sum–完整指南

> 原文：<https://www.askpython.com/python/examples/numpy-sum>

你好，欢迎来到这个关于 **Numpy 求和法**的教程。在本教程中，我们将学习 NumPy sum 方法，也将看到许多相同的例子。让我们开始吧！

***也读:[【NumPy Cos——完全指南](https://www.askpython.com/python-modules/numpy/numpy-cos)***

* * *

## 什么是 NumPy Sum？

NumPy 中的 sum 方法是一个返回数组总和的函数。它可以是整个数组的总和、沿行总和或沿列总和。我们将在本教程的下一节看到每个例子。

***亦读:[【Numpy 罪恶-完全指南】](https://www.askpython.com/python-modules/numpy/numpy-sin)***

* * *

## NumPy 和的语法

让我们首先看看 NumPy sum 函数的语法。

```py
numpy.sum(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 要求和的元素。 | 需要 |
| 轴 | 要对数组求和的轴。它可以是 axis=0，即沿列，或者 axis=1，即沿行，或者 axis=None，这意味着对整个数组求和。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| keepdims (bool) | 如果设置为真，减少的轴将作为尺寸为 1 的尺寸留在结果中。使用此选项，结果将根据输入数组正确传播。 | 可选择的 |
| 最初的 | 总和的起始值。 | 可选择的 |
| 在哪里 | 要包含在总和中的元素。 | 可选择的 |

**返回:**
一个与 *a* 形状相同的数组，包含沿给定轴的和，并移除指定轴。如果 axis=None，则返回一个标量，它是整个数组的总和。

* * *

## Numpy.sum()方法的示例

现在让我们开始使用 numpy.sum 方法，这样我们可以理解输出。

### 整个数组的 Numpy.sum()

**一维数组**

```py
import numpy as np

a = [2, 5, 3, 8, 4]

sum = np.sum(a)
print("a =", a)
print("Sum of the array =", sum)

```

**输出:**

```py
a = [2, 5, 3, 8, 4]
Sum of the array = 22

```

数组之和= 2+5+3+8+4 = 17。

**二维数组**

```py
import numpy as np

a = [[2, 5, 4], [3, 2, 1]]

sum = np.sum(a)
print("a =", a)
print("Sum of the array =", sum)

```

**输出:**

```py
a = [[2, 5, 4], [3, 2, 1]]
Sum of the array = 17

```

数组之和= 2+5+4+3+2+1 = 17

* * *

### 沿轴的 Numpy.sum()

**按列求和**

```py
import numpy as np

a = [[2, 5, 4],
     [3, 2, 1]]

# sum along axis=0 i.e. columns
sum = np.sum(a, axis=0)
print("a =", a)
print("Sum of the array =", sum)

```

**输出:**

```py
a = [[2, 5, 4], [3, 2, 1]]
Sum of the array = [5 7 5]

```

第 0 列总和= 2+3 = 5
第 1 列总和= 5+2 = 7
第 2 列总和= 4+1 = 5

**逐行求和**

```py
import numpy as np

a = [[2, 5, 4],
     [3, 2, 1]]

# sum along axis=1 i.e. rows
sum = np.sum(a, axis=1)
print("a =", a)
print("Sum of the array =", sum)

```

**输出:**

```py
a = [[2, 5, 4], [3, 2, 1]]
Sum of the array = [11  6]

```

第 0 行总和= 2+5+4 = 11
第 1 行总和= 3+2+1 = 6

* * *

### 空数组的 Numpy.sum()

```py
import numpy as np

a = []
b = [[]]

sum_a = np.sum(a)
print("a =", a)
print("Sum of the 1-d empty array =", sum_a)

sum_b = np.sum(b)
print("b =", b)
print("Sum of the 2-d empty array =", sum_b)

```

**输出:**

```py
a = []
Sum of the 1-d empty array = 0.0
b = [[]]
Sum of the 2-d empty array = 0.0

```

空数组的和是中性元素，即 0。

* * *

## 以浮点数据类型返回数组的 Numpy.sum()

这与上面的例子相同，只是这里返回值是浮点数据类型。

**整个数组的总和**

```py
import numpy as np

a = [[3, 12, 4], [3, 5, 1]]

sum = np.sum(a, dtype=float)
print("a =", a)
print("Sum of the array =", sum)

```

**输出:**

```py
a = [[3, 12, 4], [3, 5, 1]]
Sum of the array = 28.0

```

**按列求和**

```py
import numpy as np

a = [[3, 12, 4], 
     [3, 5, 1]]

# sum along axis=0 i.e. columns
sum = np.sum(a, dtype=float, axis=0)
print("a =", a)
print("Sum of the array =", sum)

```

**输出:**

```py
a = [[3, 12, 4], [3, 5, 1]]
Sum of the array = [ 6\. 17\.  5.]

```

**逐行求和**

```py
import numpy as np

a = [[3, 12, 4], 
     [3, 5, 1]]

# sum along axis=1 i.e. rows
sum = np.sum(a, dtype=float, axis=1)
print("a =", a)
print("Sum of the array =", sum)

```

**输出:**

```py
a = [[3, 12, 4], [3, 5, 1]]
Sum of the array = [19\.  9.]

```

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy sum** 方法，并使用该方法练习了不同类型的示例。

* * *

## 参考

*   [NumPy sum 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)