# NumPy nancumprod–完整指南

> 原文:1230【https://www . aspython . com/python-modules/num py/numpy-nanumprod】

大家好，欢迎来到这个关于 **Numpy nancumprod** 的教程。在我们之前的教程中，我们学习了 [NumPy cumprod](https://www.askpython.com/python-modules/numpy/numpy-cumprod) 。在本教程中，我们将学习 **NumPy nancumprod()** 方法，也将看到许多关于相同的例子。让我们开始吧！

***推荐阅读:[NumPy cum prod–完整指南](https://www.askpython.com/python-modules/numpy/numpy-cumprod)*T5、***[NumPy nan prod–完整指南](https://www.askpython.com/python-modules/numpy/numpy-nanprod)*****

* * *

## 什么是 NumPy nancumprod？

在 Python 中， **NaN** 表示**而不是数字**。如果我们有一个包含一些 NaN 值的数组，并且想要找到它的累积积，我们可以使用 NumPy 的`nancumprod()`方法。**累积积**是给定序列的部分积的序列。如果 ***{a，b，c，d，e，f，…..}*** 是一个序列那么它的累积积表示为 ***{a，a*b，a*b*c，a*b*c*d，…。}*** 。

NumPy 中的`nancumprod()`方法是一个函数，它返回通过将数组中的 NaN 值视为等于 1 而计算的数组元素的累积积。它可以是展平数组的累积积、沿行数组元素的累积积或沿列数组元素的累积积。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy nancumprod 的语法

```py
numpy.nancumprod(a, axis=None, dtype=None, out=None)

```

| **参数** | **描述** | **必需/可选** |
| a | 输入数组。 | 需要 |
| 轴 | 要沿其计算数组累积积的轴。它可以是 axis=0 或 axis=1 或 axis=None，这意味着要返回展平数组的累积积。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状和长度。 | 可选择的 |

**返回:**
一个包含输出的新数组。如果提到了 *out* ，则返回对它的引用。

* * *

## numpy.nancumprod()方法的示例

现在让我们借助一些例子来看看如何使用这个函数。

### 单个元素的累积积

```py
import numpy as np

a = 10
ans = np.nancumprod(a)
print("a =", a)
print("Cumulative product =", ans)

```

**输出:**

```py
a = 10
Cumulative product = [10]

```

* * *

### 包含 nan 的一维数组的累积积

```py
import numpy as np

a = [5, np.nan, 10, np.nan, np.nan, 2]
ans = np.nancumprod(a)
print("a =", a)
print("Cumulative product =", ans)

```

**输出:**

```py
a = [5, nan, 10, nan, nan, 2]
Cumulative product = [  5\.   5\.  50\.  50\.  50\. 100.]

```

在上面的代码中，数组包含 3 个 NaN 值。

在计算乘积时，`nancumprod()`函数将 NaN 值视为 1，并将累积乘积计算为 5，5*，5*1*10，5*1*10*1，5*1*10*1，5 * 1 * 10 * 1 * 2，结果为 5，5，50，50，50，100。

* * *

### 包含 nan 的二维数组的累积积

```py
import numpy as np

a = [[3, np.nan, 6], [8, np.nan, np.nan]]
ans = np.nancumprod(a)
print("a =", a)
print("Cumulative product =", ans)

```

**输出:**

```py
a = [[3, nan, 6], [8, nan, nan]]
Cumulative product = [  3\.   3\.  18\. 144\. 144\. 144.]

```

在二维数组的情况下，当没有提到轴时，数组首先被展平，然后通过将 NaNs 视为 1 来计算其累积积。

在上面的示例中，数组首先被展平为[3，np.nan，6，8，np.nan，np.nan]，即按行排列，然后其累积积被计算为[3，3*1，3*1*6，3*1*6*8，3*1*6*8，*1，3*1*6*8，*1*1]，这产生了函数返回的数组[3，3，18，144，144，144]。

* * *

### 将 NaN 视为 1 的沿轴累积乘积

**轴= 0**

```py
import numpy as np

a = [[5, 2, np.nan], [10, np.nan, 3]]
# cumulative product along axis=0
ans = np.nancumprod(a, axis=0)
print("a =\n", a)
print("Cumulative product =\n", ans)

```

**输出:**

```py
a =
 [[5, 2, nan], [10, nan, 3]]
Cumulative product =
 [[ 5\.  2\.  1.]
 [50\.  2\.  3.]]

```

把南当 1，第一排照原样。第二行包含计算为 5*10、2*1、1*3 的累积积，即 50、2 和 3。也就是说，累积积是按列计算的，并以行的形式存储。

**轴= 1**

```py
import numpy as np

a = [[5, 2, np.nan], [10, np.nan, 3]]
# cumulative product along axis=1
ans = np.nancumprod(a, axis=1)
print("a =\n", a)
print("Cumulative product =\n", ans)

```

**输出:**

```py
a =
 [[5, 2, nan], [10, nan, 3]]
Cumulative product =
 [[ 5\. 10\. 10.]
 [10\. 10\. 30.]]

```

这里，第一列是原样，第二列包含计算为 5*2，10*1 的累积积，得到 10，10，第三列具有 5*2*1，10*1*3 的累积积，即 10 和 30。也就是说，累积积是按行计算的，并以列的形式存储。

* * *

## 摘要

仅此而已！在本教程中，我们学习了 **Numpy nancumprod** 方法，并使用该方法练习了不同类型的示例。

* * *

## 参考

*   [NumPy nancumprod 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.nancumprod.html)