# NumPy ones _ like 完整指南

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-ones_like>

在本教程中，我们将学习到 **NumPy ones_like** 方法，也将看到许多关于相同的例子。让我们开始吧！

***推荐阅读:***

* * *

## NumPy ones _ 是什么样子的？

NumPy 中的`ones_like`方法是一个函数，它返回与给定数组具有相同形状和大小的一个数组。

* * *

## NumPy ones _ like 的语法

让我们先来看看`numpy.ones_like()`方法的语法。

```py
numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 定义要返回的数组的形状和数据类型的对象。 | 需要 |
| 数据类型 | 所需数组的数据类型。覆盖结果的数据类型。 | 可选择的 |
| 命令 | 多维数据在存储器中存储的期望顺序。它可以是 row-major ('C ')，column-major ('F ')，如果 *a* 是 Fortran 连续的，则“A”表示“F”，否则为“C”。‘k’表示尽可能匹配 *a* 的布局。 | 可选择的 |
| subok (bool) | 确定新创建的数组是使用子类类型 *a* (subok=True)还是基类数组(subok=False)。
默认值为**真**。 | 可选择的 |
| 形状 | 所需数组的形状。覆盖结果的形状。 | 可选择的 |

**返回:**
与给定数组形状和数据类型相同的数组，用全 1 填充。

* * *

## Numpy ones _ like 函数的示例

现在让我们看看`numpy.ones_like()`函数是如何工作的，以及不同类型的输入的预期输出是什么。

### 使用 ones _ like 的一维数组

```py
import numpy as np

a = np.arange(10)
print("a =", a)

b = np.ones_like(a)
print("b =", b)

```

**输出:**

```py
a = [0 1 2 3 4 5 6 7 8 9]
b = [1 1 1 1 1 1 1 1 1 1]

```

* * *

### 使用 ones _ like 的二维数组

**N×N 阵列**

```py
import numpy as np

a = np.arange(10).reshape(5, 2)
print("a =\n", a)

b = np.ones_like(a)
print("b =\n", b)

```

**输出:**

```py
a =
 [[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
b =
 [[1 1]
 [1 1]
 [1 1]
 [1 1]
 [1 1]]

```

**1×N 阵列**

```py
import numpy as np

a = np.arange(12).reshape(1, 12)
print("a =\n", a)

b = np.ones_like(a)
print("b =\n", b)

```

**输出:**

```py
a =
 [[ 0  1  2  3  4  5  6  7  8  9 10 11]]
b =
 [[1 1 1 1 1 1 1 1 1 1 1 1]]

```

**N×1 阵列**

```py
import numpy as np

a = np.arange(12).reshape(12, 1)
print("a =\n", a)

b = np.ones_like(a)
print("b =\n", b)

```

**输出:**

```py
a =
 [[ 0]
 [ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]
 [11]]
b =
 [[1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]
 [1]]

```

* * *

### 使用 Numpy ones _ like 的一维浮点型数组

```py
import numpy as np

a = np.arange(10)
print("a =", a)

b = np.ones_like(a, dtype=float)
print("b =", b)

```

**输出:**

```py
a = [0 1 2 3 4 5 6 7 8 9]
b = [1\. 1\. 1\. 1\. 1\. 1\. 1\. 1\. 1\. 1.]

```

* * *

### 二维浮点型数组

```py
import numpy as np

a = np.arange(10).reshape(2, 5)
print("a =\n", a)

b = np.ones_like(a, dtype=float)
print("b =\n", b)

```

**输出:**

```py
a =
 [[0 1 2 3 4]
 [5 6 7 8 9]]
b =
 [[1\. 1\. 1\. 1\. 1.]
 [1\. 1\. 1\. 1\. 1.]]

```

* * *

## Numpy 和 ones _ like 有什么区别

*   注意，在`ones`方法中，我们正在创建一个我们想要的形状和数据类型的新数组，所有的值都是 1。但是，在这里，我们直接传递一个数组或类似数组的对象来获得一个具有相同形状和数据类型的数组。
*   NumPy `ones_like`函数比 NumPy `ones`函数花费更多的时间来生成全 1 数组。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy ones_like** 方法，并使用相同的方法练习了不同类型的示例。如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy one _ like 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html)