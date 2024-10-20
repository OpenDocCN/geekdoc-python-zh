# NumPy cum prod–完整指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-cumber

你好，欢迎来到这个关于 **Numpy cumprod** 的教程。在本教程中，我们将学习 NumPy `cumprod()`方法，也将看到许多关于这个方法的例子。让我们开始吧！

***亦读:[NumPy cumsum——完全指南](https://www.askpython.com/python-modules/numpy/numpy-cumsum)***

* * *

## 什么是 NumPy cumprod？

**累积积**是给定序列的部分积的序列。如果 ***{a，b，c，d，e，f，…..}*** 是一个序列那么它的累积积表示为 ***{a，a*b，a*b*c，a*b*c*d，…。}*** 。

NumPy 中的`cumprod()`方法返回沿着指定轴的输入数组元素的累积积。它可以是展平数组的累积积、沿行数组元素的累积积或沿列数组元素的累积积。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy cumprod 的语法

```py
numpy.cumprod(a, axis=None, dtype=None, out=None)

```

| **参数** | **描述** | **必需/可选** |
| a | 输入数组。 | 需要 |
| 轴 | 要沿其计算数组累积积的轴。它可以是 axis=0，即沿列，也可以是 axis=1，即沿行，或者 axis=None，这意味着要返回展平数组的累积积。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状和长度。 | 可选择的 |

**返回:**
一个包含输出的新数组。如果提到了 *out* ，则返回对它的引用。

* * *

## numpy.cumprod 方法的示例

现在让我们开始使用 numpy.cumprod 方法，这样我们就可以理解输出。

### 单个元素的累积积

```py
import numpy as np

a = 5
ans = np.cumprod(a)

print("a =", a)
print("Cumulative product =", ans)

```

**输出:**

```py
a = 5
Cumulative product = [5]

```

* * *

### 空数组的累积积

```py
import numpy as np

a = []
ans = np.cumprod(a)

print("a =", a)
print("Cumulative product =", ans)

```

**输出:**

```py
a = []
Cumulative product = []

```

* * *

### 一维数组的累积积

```py
import numpy as np

a = [2, 10, 3 ,6]
ans = np.cumprod(a)

print("a =", a)
print("Cumulative product of the array =", ans)

```

**输出:**

```py
a = [2, 10, 3, 6]
Cumulative product of the array = [  2  20  60 360]

```

这里，累积积计算为 2，2*10，2*10*3，2*10*3*6 即 2，20，60，360。

* * *

### 二维数组的累积积

```py
import numpy as np

a = [[8, 3], [5, 2]]
ans = np.cumprod(a)

print("a =", a)
print("Cumulative product of the array =", ans)

```

**输出:**

```py
a = [[8, 3], [5, 2]]
Cumulative product of the array = [  8  24 120 240]

```

在二维数组的情况下，当没有提到轴时，数组首先被展平，然后计算其累积积。
在上面的示例中，数组首先被展平为[8，3，5，2]，即按行排列，然后其累积积被计算为[8，8*3，8*3*5，8*3*5*2]，这产生了函数返回的数组[8，24，120，240]。

* * *

### 以浮点数据类型返回数组的 Numpy.cumprod()

这与上面的例子相同，只是这里返回值是浮点数据类型。

```py
import numpy as np

a = [2, 10, 3, 6]
ans = np.cumprod(a, dtype=float)

print("a =", a)
print("Cumulative product of the array =", ans)

```

**输出:**

```py
a = [2, 10, 3, 6]
Cumulative product of the array = [  2\.  20\.  60\. 360.]

```

* * *

### 沿着轴的累积乘积

**轴= 0**

```py
import numpy as np

a = [[3, 2, 1], [4, 5, 6]]
# cumulative product along axis=0
ans = np.cumprod(a, axis=0)

print("a =\n", a)
print("Cumulative product of the array =\n", ans)

```

**输出:**

```py
a =
 [[3, 2, 1], [4, 5, 6]]
Cumulative product of the array =
 [[ 3  2  1]
 [12 10  6]]

```

这里，第一行是原样，第二行包含计算为 3*4、2*5、1*6 的累积积，结果是 12、10 和 6。

**轴= 1**

```py
import numpy as np

a = [[3, 2, 1], [4, 5, 6]]
# cumulative product along axis=1
ans = np.cumprod(a, axis=1)

print("a =\n", a)
print("Cumulative product of the array =\n", ans)

```

**输出:**

```py
a =
 [[3, 2, 1], [4, 5, 6]]
Cumulative product of the array =
 [[  3   6   6]
 [  4  20 120]]

```

这里，第一列是原样，第二列包含计算为 3*2，4*5 的累积积，得到 6，20，第三列具有 3*2*1，4*5*6 的累积积，即 6 和 120。

* * *

## 摘要

仅此而已！在本教程中，我们学习了 **Numpy cumprod** 方法，并使用相同的方法练习了不同类型的示例。

* * *

## 参考

*   [NumPy cumprod 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html)