# numpy 纳米棒–完整指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-nanprod

你好，欢迎来到这个关于 Numpy nanprod 的教程。在本教程中，我们将学习 NumPy nanprod()方法，也将看到许多相同的例子。让我们开始吧！

***也读作:[【Numpy trunc()——返回输入的截断值，逐元素](https://www.askpython.com/python-modules/numpy/numpy-trunc)***

* * *

## 什么是 NumPy nanprod？

在 Python 中，NaN 表示的不是数字。如果我们有一个包含一些 NaN 值的数组，并且想要找到它的乘积，我们可以使用 NumPy 的`nanprod()`方法。

NumPy 中的`nanprod()`方法是一个函数，它返回通过将数组中的 NaN 值视为等于 1 而计算的数组元素的乘积。它可以是所有数组元素的乘积、沿行数组元素的乘积或沿列数组元素的乘积。

我们将在本教程的下一节看到每个例子。

* * *

## 纳米棒语法

```py
numpy.nanprod(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 需要其产品的输入数组。 | 需要 |
| 轴 | 沿其计算数组乘积的轴。它可以是 axis=0，即沿列，也可以是 axis=1，即沿行，或者 axis=None，这意味着要返回整个数组的乘积。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| keepdims (bool) | 如果设置为真，减少的轴将作为尺寸为 1 的尺寸留在结果中。使用此选项，结果将根据输入数组正确传播。 | 可选择的 |
| 最初的 | 产品的起始值。 | 可选择的 |
| 在哪里 | 产品中包含的元素。 | 可选择的 |

**返回:**
一个与 *a* 形状相同的数组，该数组包含 *a* 的元素的乘积，将 NaN 值视为 1，沿给定的轴并移除指定的轴。如果 axis=None，则返回一个标量，它是整个数组的乘积。

* * *

## numpy.nanprod()的示例

`numpy.nanprod()`函数用于计算给定轴上数组元素的乘积，忽略 nan。让我们通过一些例子来看看`numpy.nanprod()`的用法。

### 使用 numpy.nanprod()的整个数组的乘积

**一维数组**

```py
import numpy as np

a = np.array([6, np.nan, 7])
product = np.nanprod(a)
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [ 6\. nan  7.]
Product of the array = 42.0

```

在上面的代码中，数组包含一个 NaN 值。在计算乘积时，`nanprod()`函数将 NaN 值视为 1，并将乘积计算为 6*1*7 = 42。

**二维数组**

```py
import numpy as np

a = np.array([[6, np.nan, 7], [np.nan, np.nan, 3]])
product = np.nanprod(a)
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [[ 6\. nan  7.]
 [nan nan  3.]]
Product of the array = 126.0

```

将所有 NaN 值视为 1，乘积= 6*1*7*1*1*3 = 126。

* * *

### 沿着轴的产品

**列式产品**

```py
import numpy as np

a = np.array([[np.nan, np.nan, 4],
              [5, np.nan, 10]])

# product along axis=0 i.e. columns
product = np.nanprod(a, axis=0)
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [[nan nan  4.]
 [ 5\. nan 10.]]
Product of the array = [ 5\.  1\. 40.]

```

将 NaN 值视为 1，
列 0 乘积= 1*5 = 5
列 1 乘积= 1*1 = 1
列 2 乘积= 4*10 = 40

**逐行乘积**

```py
import numpy as np

a = np.array([[np.nan, np.nan, 4],
              [5, np.nan, 10]])

# product along axis=1 i.e. rows
product = np.nanprod(a, axis=1)
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [[nan nan  4.]
 [ 5\. nan 10.]]
Product of the array = [ 4\. 50.]

```

将 NaN 值视为 1，
第 0 行乘积= 1*1*4 = 4
第 1 行乘积= 5*1*10 = 50

* * *

## 空数组和全 NaN 数组的乘积

```py
import numpy as np

# empty arrays
a = []
b = [[]]

product_a = np.nanprod(a)
print("a =", a)
print("Product of the 1-d empty array =", product_a)

product_b = np.nanprod(b)
print("b =", b)
print("Product of the 2-d empty array =", product_b)

# all NaN array
c = [np.nan, np.nan, np.nan]
product_c = np.nanprod(c)
print("c =", c)
print("Product of the all NaN array =", product_c)

```

**输出:**

```py
a = []
Product of the 1-d empty array = 1.0
b = [[]]
Product of the 2-d empty array = 1.0
c = [nan, nan, nan]
Product of the all NaN array = 1.0

```

当对所有空数组和只包含 NaN 值的数组应用`nanprod()`方法时，它们返回 1。

* * *

## 结论

仅此而已！在本教程中，我们学习了 Numpy nanprod 方法，并使用该方法练习了不同类型的示例。

* * *

## 参考

*   [NumPy nanprod 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html)