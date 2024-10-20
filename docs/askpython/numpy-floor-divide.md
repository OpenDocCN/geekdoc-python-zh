# NumPy floor _ divide–完整指南

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-floor-divide>

大家好，欢迎来到这个关于 **Numpy floor_divide** 的教程。在本教程中，我们将学习 **NumPy floor_divide()** 方法，也将看到许多关于相同的例子。让我们开始吧！

***推荐阅读——[Numpy 楼](https://www.askpython.com/python-modules/numpy/numpy-floor)***

* * *

## 什么是 NumPy floor_divide？

Python 中的 floor 运算符用`//`表示。NumPy `floor_divide`运算符是`//`和`%`的组合，然后开始余数或 mod 运算符。简化的方程式是`a = a%b + b*(a//b)`。

NumPy `floor_divide()`方法返回小于或等于输入除法的最大整数。输入可以是两个数组，也可以是一个数组和一个标量。

我们将在本教程的下一节看到每个例子。

* * *

## NumPy floor_divide 的语法

```py
numpy.floor_divide(x1, x2, out=None)

```

| **参数** | **描述** | **必需/可选** |
| x1 | 输入(分子) | 需要 |
| x2 | 输入(分母) | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状和长度。 | 可选择的 |

**返回:**
一个 n 维数组 *y* ，`y=(x1/x2)`。如果 *x1* 和 *x2* 都是标量，那么 *y* 就是标量。

* * *

## NumPy floor_divide 示例

现在让我们看几个例子来更好地理解这个函数。

### 当除数是标量时

**一维数组**

```py
import numpy as np 

a = 5
arr = [12, 15, 10, 6]
ans = np.floor_divide(arr, a)
print("a =", a)
print("Result =", ans)

```

**输出:**

```py
a = 5
Result = [2 3 2 1]

```

`floor_divide`方法通过将列表中的每个元素除以 *a* 并计算其最低值来计算输出。这里，12/5 = 2.4，floor(2.4)= 2；15/5 = 3，楼层(3)= 3；10/5 = 2，楼层(2) = 2，6/5 = 1.2，楼层(1.2) = 1。
因此，得到的数组是[2，3，2，1]。

//运算符可以用作 ndarrays 上 numpy.floor_divide 的简写。

```py
import numpy as np 

a = 5
arr = np.array([12, 15, 10, 6])
ans = arr // a
print("a =", a)
print("Result =", ans)

```

**输出:**

```py
a = 5
Result = [2 3 2 1]

```

**二维数组**

```py
import numpy as np 

a = 5
arr = [[12, 15], [0, 36]]
ans = np.floor_divide(arr, a)
print("a =", a)
print("Result =\n", ans)

```

**输出:**

```py
a = 5
Result =
 [[2 3]
 [0 7]]

```

与一维数组类似，这里每个元素都除以 *a* ，然后计算其底值，并存储在结果数组中。

*   结果[0][0] =下限(12/5) =下限(2.4) = 2
*   结果[0][1] =下限(15/5) =下限(3) = 3
*   结果[1][0] =下限(0/5) =下限(0) = 0
*   结果[1][1] =下限(36/5) =下限(7.2) = 7

* * *

### 2 个数组/列表的除法

```py
import numpy as np

arr1 = [10, 20, 30, 40]
arr2 = [4, 3, 6, 5]
ans = np.floor_divide(arr1, arr2)
print("Array 1 =", arr1)
print("Array 2 =", arr2)
print("Result =", ans)

```

**输出:**

```py
Array 1 = [10, 20, 30, 40]
Array 2 = [4, 3, 6, 5]
Result = [2 6 5 8]

```

在这种情况下，对两个数组中的相应元素执行除法和取整运算。

*   结果[0] =arr1[0]/arr2[0] =下限(10/4) =下限(2.5) = 2
*   结果[1] =arr1[1]/arr2[1] =下限(20/3) =下限(6.666) = 6
*   结果[2] =arr1[2]/arr2[2] =下限(30/6) =下限(5) = 5
*   结果[3] =arr1[3]/arr2[3] =下限(40/5) =下限(8) = 8

* * *

### 当数组包含负元素时

到目前为止，我们已经看到了所有元素都是正数的例子。现在让我们看看`numpy.floor_divide()`方法如何处理负值。

```py
import numpy as np

arr1 = [16, 5, -30, 18]
arr2 = [8, -5, 7, 10]
ans = np.floor_divide(arr1, arr2)
print("Array 1 =", arr1)
print("Array 2 =", arr2)
print("Result =", ans)

```

**输出:**

```py
Array 1 = [16, 5, -30, 18]
Array 2 = [8, -5, 7, 10]
Result = [ 2 -1 -5  1]

```

*   结果[0]= arr 1[0]/arr 2[0]= floor(16/8)= floor(2)= 2
*   结果[1]= arr 1[1]/arr 2[1]= floor(5/-5)= floor(-1)=-1
*   结果[2] =arr1[2]/arr2[2] =下限(-30/7) =下限(-4.2857) = -5
*   结果[3] =arr1[3]/arr2[3] =下限(18/10) =下限(1.8) = 1

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy floor_divide** 方法，并使用该方法练习了不同类型的示例。如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy floor_divide 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html)