# NumPy cumsum–完整指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-cumsum

你好，欢迎来到这个关于 **Numpy cumsum** 的教程。在本教程中，我们将学习 NumPy `cumsum()`方法，也将看到许多关于这个方法的例子。让我们开始吧！

* * *

## 什么是 NumPy cumsum？

**累积和**是给定序列的部分和的序列。如果 ***{a，b，c，d，e，f，…..}*** 是一个序列那么它的累加和表示为 ***{a，a+b，a+b+c，a+b+c+d，…。}*** 。

NumPy 中的`cumsum()`方法返回沿着指定轴的输入数组元素的累积和。它可以是展平数组的累积和、沿行数组元素的累积和或沿列数组元素的累积和。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy cumsum 的语法

```py
numpy.cumsum(a, axis=None, dtype=None, out=None)

```

| **参数** | **描述** | **必需/可选** |
| a | 输入数组。 | 需要 |
| 轴 | 要沿其计算数组累积和的轴。它可以是 axis=0，即沿列，也可以是 axis=1，即沿行，或者 axis=None，这意味着要返回展平数组的累积和。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状和长度。 | 可选择的 |

**返回:**
一个包含输出的新数组。如果提到了 *out* ，则返回对它的引用。

* * *

## 例子

现在让我们开始使用 numpy.cumsum 方法，这样我们就可以理解输出。

### 单个元素的累积和

```py
import numpy as np

a = 5
ans = np.cumsum(a)

print("a =", a)
print("Cumulative sum =", ans)

```

**输出:**

```py
a = 5
Cumulative sum = [5]

```

* * *

### 空数组的累积和

```py
import numpy as np

a = []
ans = np.cumsum(a)

print("a =", a)
print("Cumulative sum =", ans)

```

**输出:**

```py
a = []
Cumulative sum = []

```

* * *

### 一维数组的 Numpy 累积和

```py
import numpy as np

a = [5, 3, 2, 8]
ans = np.cumsum(a)

print("a =", a)
print("Cumulative sum of the array =", ans)

```

**输出:**

```py
a = [5, 3, 2, 8]
Cumulative sum of the array = [ 5  8 10 18]

```

这里，累积和的计算方式为 5，5+3，5+3+2，5+3+2+8，结果为 5，8，10，18。

* * *

### 二维数组的 Numpy 累积和

```py
import numpy as np

a = [[4, 3], [9, 10]]
ans = np.cumsum(a)

print("a =", a)
print("Cumulative sum of the array =", ans)

```

**输出:**

```py
a = [[4, 3], [9, 10]]
Cumulative sum of the array = [ 4  7 16 26]

```

在二维数组的情况下，当没有提到轴时，数组首先被展平，然后计算其累积和。
在上面的示例中，数组首先被展平为[4，3，9，10]，即按行排列，然后其累积和被计算为[4，4+3，4+3+9，4+3+9+10]，这产生了函数返回的数组[4，7，16，26]。

* * *

### 以浮点数据类型返回数组的 Numpy.cumsum()

这与上面的例子相同，只是这里返回值是浮点数据类型。

```py
import numpy as np

a = [5, 3, 2, 8]
ans = np.cumsum(a, dtype=float)

print("a =", a)
print("Cumulative sum of the array =", ans)

```

**输出:**

```py
a = [5, 3, 2, 8]
Cumulative sum of the array = [ 5\.  8\. 10\. 18.]

```

* * *

## 沿轴累计总和

**轴= 0**

```py
import numpy as np

a = [[1, 5, 3], [7, 10, 4]]
# cumulative sum along axis=0
ans = np.cumsum(a, axis=0)

print("a =\n", a)
print("Cumulative sum of the array =\n", ans)

```

**输出:**

```py
a =
 [[1, 5, 3], [7, 10, 4]]
Cumulative sum of the array =
 [[ 1  5  3]
 [ 8 15  7]]

```

这里，第一行是原样，第二行包含按 1+7、5+10 和 3+4 计算的累积和，结果是 8、15 和 7。

**轴= 1**

```py
import numpy as np

a = [[1, 5, 3], [7, 10, 4]]
# cumulative sum along axis=1
ans = np.cumsum(a, axis=1)

print("a =\n", a)
print("Cumulative sum of the array =\n", ans)

```

**输出:**

```py
a =
 [[1, 5, 3], [7, 10, 4]]
Cumulative sum of the array =
 [[ 1  6  9]
 [ 7 17 21]]

```

这里，第一列是原样，第二列包含计算为 1+5，7+10 的累积和，结果是 6，17，第三列具有 1+5+3，7+10+4 的累积和，即 9 和 21。

* * *

## 摘要

仅此而已！在本教程中，我们学习了 **Numpy cumsum** 方法，并使用相同的方法练习了不同类型的示例。

* * *

## 参考

*   [NumPy cumsum 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)