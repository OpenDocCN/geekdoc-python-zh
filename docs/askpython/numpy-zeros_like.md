# NumPy zeros _ like–完整指南

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-zeros_like>

在本教程中，我们将学习 NumPy zeros_like 方法，也将看到许多关于这个方法的例子。让我们开始吧！

***推荐阅读:[【NumPy 零点——完整指南](https://www.askpython.com/python/numpy-zeros)***

* * *

## NumPy zeros_like 是什么？

NumPy 中的`zeros_like`方法是一个函数，它返回与给定数组具有相同形状和大小的零数组。

* * *

## 类似零的语法

**numpy。** **零 _ 像** **(** *一* **，***dtype =无* **，** *order='K'* **，***subok =真* **，***shape =无* **)**

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 定义要返回的数组的形状和数据类型的对象。 | 需要 |
| 数据类型 | 所需数组的数据类型。覆盖结果的数据类型。 | 可选择的 |
| 命令 | 多维数据在存储器中存储的期望顺序。它可以是 row-major ('C ')，column-major ('F ')，如果 *a* 是 Fortran 连续的，则“A”表示“F”，否则为“C”。‘k’表示尽可能匹配 *a* 的布局。 | 可选择的 |
| subok (bool) | 确定新创建的数组是使用子类类型 *a* (subok=True)还是基类数组(subok=False)。
默认值为**真**。 | 可选择的 |
| 形状 | 所需数组的形状。覆盖结果的形状。 | 可选择的 |

**返回:**
与给定数组具有相同形状和数据类型的数组。

* * *

## Numpy zeros_like 函数示例

现在让我们看看 numpy.zeros_like()函数是如何工作的，以及不同类型的输入的预期输出是什么。

### 使用 zeros_like 的一维数组

```py
import numpy as np

a = np.arange(10)
print("a =", a)

b = np.zeros_like(a)
print("b =", b)

```

**输出:**

```py
a = [0 1 2 3 4 5 6 7 8 9]
b = [0 0 0 0 0 0 0 0 0 0]

```

* * *

### 使用 zeros_like 的二维数组

**N×N 阵列**

```py
import numpy as np

a = np.arange(10).reshape(2, 5)
print("a =\n", a)

b = np.zeros_like(a)
print("b =\n", b)

```

**输出:**

```py
a =
 [[0 1 2 3 4]
 [5 6 7 8 9]]
b =
 [[0 0 0 0 0]
 [0 0 0 0 0]]

```

**1×N 阵列**

```py
import numpy as np

a = np.arange(10).reshape(1, 10)
print("a =\n", a)

b = np.zeros_like(a)
print("b =\n", b)

```

**输出:**

```py
a =
 [[0 1 2 3 4 5 6 7 8 9]]
b =
 [[0 0 0 0 0 0 0 0 0 0]]

```

**N×1 阵列**

```py
import numpy as np

a = np.arange(10).reshape(10, 1)
print("a =\n", a)

b = np.zeros_like(a)
print("b =\n", b)

```

**输出:**

```py
a =
 [[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]
b =
 [[0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]
 [0]]

```

* * *

### 一维浮点型数组

```py
import numpy as np

a = np.arange(10)
print("a =", a)

b = np.zeros_like(a, dtype=float)
print("b =", b)

```

**输出:**

```py
a = [0 1 2 3 4 5 6 7 8 9]
b = [0\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 0\. 0.]

```

* * *

### 二维浮点型数组

```py
import numpy as np

a = np.arange(10).reshape(2, 5)
print("a =\n", a)

b = np.zeros_like(a, dtype=float)
print("b =\n", b)

```

**输出:**

```py
a =
 [[0 1 2 3 4]
 [5 6 7 8 9]]
b =
 [[0\. 0\. 0\. 0\. 0.]
 [0\. 0\. 0\. 0\. 0.]]

```

* * *

## 零和类零的区别

注意，在`zeros`方法中，我们正在创建一个我们想要的形状和数据类型的新数组，所有的值都是 0。但是，在这里，我们直接传递一个数组或类似数组的对象来获得一个具有相同形状和数据类型的数组。NumPy `zeros_like`函数比 NumPy `zeros`函数花费更多的时间来产生一个全 0 的数组。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy zeros_like** 方法，并使用该方法练习了不同类型的示例。

* * *

## 参考

*   [NumPy zeros_like 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html)