# NumPy nancumsum–完全指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-nanum

大家好，欢迎来到这个关于 **Numpy nancumsum** 的教程。在我们之前的教程中，我们已经了解了 *[NumPy cumsum](https://www.askpython.com/python-modules/numpy/numpy-cumsum)* 和 *[NumPy nansum](https://www.askpython.com/python-modules/numpy/numpy-nansum)* 。在本教程中，我们将学习 **NumPy nancumsum()** 方法，也将看到许多相同的例子。让我们开始吧！

***推荐阅读: [NumPy cumsum](https://www.askpython.com/python-modules/numpy/numpy-cumsum)*** ，***[NumPy nansum](https://www.askpython.com/python-modules/numpy/numpy-nansum)***

* * *

## 什么是 NumPy nancumsum？

在 Python 中， **NaN** 表示**而不是数字**。如果我们有一个包含一些 NaN 值的数组，并且想要找到它的累积和，我们可以使用 NumPy 的`nancumsum()`方法。

**累积和**是给定序列的部分和的序列。如果 ***{a，b，c，d，e，f，…..}*** 是一个序列那么它的累加和表示为 ***{a，a+b，a+b+c，a+b+c+d，…。}*** 。

NumPy 中的`nancumsum()`方法是一个函数，它返回通过将数组中的 NaN 值视为等于 0 而计算的数组元素的累积和。它可以是展平数组的累积和、沿行数组元素的累积和或沿列数组元素的累积和。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy nancumsum 的语法

```py
numpy.nancumsum(a, axis=None, dtype=None, out=None)

```

| **参数** | **描述** | **必需/可选** |
| a | 输入数组。 | 需要 |
| 轴 | 要沿其计算数组累积和的轴。它可以是 axis=0 或 axis=1 或 axis=None，这意味着要返回展平数组的累积和。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状和长度。 | 可选择的 |

**返回:**
一个新的数组，它包含将 NaN 值视为等于零的输出，即累积和。如果提到了 *out* ，则返回对它的引用。

* * *

## numpy.nancumsum()方法的示例

现在让我们借助一些例子来看看如何使用这个函数。

### 单个元素的累积和

```py
import numpy as np

a = 8
ans_a = np.nancumsum(a)

b = np.nan
ans_b = np.nancumsum(b)

print("a =", a)
print("Cumulative sum of a =", ans_a)

print("b =", b)
print("Cumulative sum of b =", ans_b)

```

**输出:**

```py
a = 8
Cumulative sum of a = [8]
b = nan
Cumulative sum of b = [0.]

```

* * *

### 包含 nan 的一维数组的累积和

```py
import numpy as np

arr = [7, 8, np.nan, 10, np.nan, np.nan]
ans = np.nancumsum(arr)

print("arr =", arr)
print("Cumulative sum of arr =", ans)

```

**输出:**

```py
arr = [7, 8, nan, 10, nan, nan]
Cumulative sum of arr = [ 7\. 15\. 15\. 25\. 25\. 25.]

```

在上面的代码中，数组包含 3 个 NaN 值。在计算累积和时，`nancumsum()`方法将这些值视为等于零。因此，累积和计算为 7，7+8，7+8+0，7+8+0+10，7+8+0+10+0，7+8+0+10+0+0，结果为 7，15，15，25，25，25。

* * *

### 包含 nan 的二维数组的累积和

```py
import numpy as np

arr = [[5, np.nan, 3], [np.nan, 2, 1]]
ans = np.nancumsum(arr)

print("arr =", arr)
print("Cumulative sum of arr =", ans)

```

**输出:**

```py
arr = [[5, nan, 3], [nan, 2, 1]]
Cumulative sum of arr = [ 5\.  5\.  8\.  8\. 10\. 11.]

```

在二维数组的情况下，当没有提到轴时，数组首先被展平，然后通过将 NaNs 视为 0 来计算其累积和。

在上面的示例中，数组首先被展平为[5，np.nan，3，np.nan，2，1]，即按行排列，然后其累积和被计算为[5，5+0，5+0+3+0，5+0+3+0+2，5+0+3+0+2+1]，这产生了函数返回的数组[5，5，8，8，10，11]。

* * *

### 将 NaN 视为 0 的沿轴累计总和

**轴=0**

```py
import numpy as np

arr = [[8, np.nan, 6], [np.nan, 10, 20]]
# cumulative sum along axis=0
ans = np.nancumsum(arr, axis=0)

print("arr =\n", arr)
print("Cumulative sum of arr =\n", ans)

```

**输出:**

```py
arr =
 [[8, nan, 6], [nan, 10, 20]]
Cumulative sum of arr =
 [[ 8\.  0\.  6.]
 [ 8\. 10\. 26.]]

```

把南当 0，第一排照原样。第二行包含按 8+0、0+10、6+20 计算的累积和，即 8、10 和 26。也就是说，累积和是按列计算的，并以行的形式存储。

**轴=1**

```py
import numpy as np

arr = [[8, np.nan, 6], [np.nan, 10, 20]]
# cumulative sum along axis=1
ans = np.nancumsum(arr, axis=1)

print("arr =\n", arr)
print("Cumulative sum of arr =\n", ans)

```

**输出:**

```py
arr =
 [[8, nan, 6], [nan, 10, 20]]
Cumulative sum of arr =
 [[ 8\.  8\. 14.]
 [ 0\. 10\. 30.]]

```

这里，第一列是原样，第二列包含计算为 8+0，0+10 的累积和，结果是 8，10，第三列具有 8+0+6，0+10+20 的累积和，即 14 和 30。也就是说，累积和是按行计算的，并以列的形式存储。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy nancumsum** 方法，并使用该方法练习了不同类型的示例。你可以从我们的 NumPy 教程[这里](https://www.askpython.com/python-modules/numpy)了解更多关于 NumPy 的信息。

* * *

## 参考

*   [NumPy nancumsum 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.nancumsum.html)