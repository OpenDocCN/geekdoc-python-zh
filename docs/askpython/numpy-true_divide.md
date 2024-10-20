# NumPy true _ Divide–按参数划分元素

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-true_divide>

大家好，欢迎来到这个关于 **Numpy true_divide** 的教程。在本教程中，我们将学习 **NumPy true_divide()** 方法，也将看到许多关于这个方法的例子。让我们开始吧！

***亦读:[【NumPy floor _ divide——完整指南](https://www.askpython.com/python-modules/numpy/numpy-floor-divide)***

* * *

## 什么是 NumPy true_divide？

`true_divide()`是 [NumPy](https://www.askpython.com/python-modules/numpy) 中的一个函数，它将一个数组中的元素按元素方式除以另一个数组中的元素，并返回一个包含答案的数组，即每个元素方式除法的商。

如果 *x1* 和 *x2* 是两个数组，那么`true_divide(x1, x2)`将执行元素除法，使得 *x1* 中的每个元素除以 *x2* 中的相应元素，并将结果存储在一个新数组中。

* * *

## NumPy true_divide 的语法

让我们来看看`true_divide()`函数的语法。

```py
numpy.true_divide(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

```

| **参数** | **描述** | **必需/可选** |
| x1 (array_like) | 红利数组。 | 需要 |
| x2(类似数组) | 除数数组。 | 需要 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| 在哪里 | 接受一个类似数组的对象。在为真的位置，`out`数组将被设置为`ufunc`结果。在其他地方，`out`数组将保持其原始值。 | 可选择的 |

**返回:**
一个数组，包含 *x1* 除以 *x2* 的按元素划分的商。
如果两个输入都是标量，那么结果也是标量。

* * *

## floor_divide 和 true_divide 的区别

在 Python 中，Numpy `floor_divide()`函数按元素执行两个数组的下限除法。它相当于使用了`/`操作符。

然而，NumPy **`true_divide()`** 按元素执行除法，相当于使用`//`运算符。

* * *

## NumPy true_divide 的示例

现在让我们进入例子，理解`true_divide`方法实际上是如何工作的。

### 当两个输入都是标量时

```py
import numpy as np 

a = 18
b = 5
c = 6
# using the true_divide function to perform element-wise division
ans_1 = np.true_divide(a, b)
ans_2 = np.true_divide(a, c)

print("a =", a, "\nb =", b)
print("Result 1 =", ans_1)
print("Result 2 =", ans_2)

```

**输出:**

```py
a = 18 
b = 5
Result 1 = 3.6
Result 2 = 3.0

```

简单的例子

```py
ans_1 = 18/5 = 3.6
ans_2 = 18/6 = 3

```

* * *

### 当一个输入是标量而另一个是一维数组时

```py
import numpy as np 

a = [2, 36, 10, 4, 20]
b = 5
# using the true_divide function to perform element-wise division
ans = np.true_divide(a, b)

print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = [2, 36, 10, 4, 20] 
b = 5
Result = [0.4 7.2 2\.  0.8 4\. ]

```

这里，标量值是除数，它将被除数数组中的每个元素划分如下:

```py
ans[0] = 2/5 = 0.4
ans[1] = 36/5 = 7.2
ans[2] = 10/5 = 2
ans[3] = 4/5 = 0.8
ans[4] = 20/5 = 4

```

* * *

### 当两个输入阵列都是一维时

```py
import numpy as np 

a = [5, 30, 12, 36, 11]
b = [2, 5, 6, 7, 10]
# using the true_divide function to perform element-wise division
ans = np.true_divide(a, b)

print("a =", a, "\nb =", b)
print("Result =", ans)

```

**输出:**

```py
a = [5, 30, 12, 36, 11] 
b = [2, 5, 6, 7, 10]
Result = [2.5        6\.         2\.         5.14285714 1.1       ]

```

这里， *a* 中的每个元素除以 *b* 中的相应元素，输出计算如下:

```py
ans[0] = a[0]/b[0] = 5/2 = 2.5
ans[1] = a[1]/b[1] = 30/5 = 6
ans[2] = a[2]/b[2] = 12/6 = 2
ans[3] = a[3]/b[3] = 36/7 = 5.14285714
ans[4] = a[4]/b[4] = 11/10 = 1.1

```

* * *

### 当两个输入阵列都是二维时

```py
import numpy as np 

a = [[25, 23], [12, 18]]
b = [[5, 6], [4, 5]]
# using the true_divide function to perform element-wise division
ans = np.true_divide(a, b)

print("a =", a, "\nb =", b)
print("Result =\n", ans)

```

**输出:**

```py
a = [[25, 23], [12, 18]] 
b = [[5, 6], [4, 5]]
Result =
 [[5\.         3.83333333]
 [3\.         3.6       ]]

```

类似于上面的例子，

```py
ans[0][0] = a[0][0]/b[0][0] = 25/5 = 5
ans[0][1] = a[0][1]/b[0][1] = 23/6 = 3.83333333

ans[1][0] = a[1][0]/b[1][0] = 12/4 = 3
ans[1][1] = a[1][1]/b[1][1] = 18/5 = 3.6

```

* * *

### true_divide (//)和 floor_divide (/)的比较

```py
import numpy as np 

a = [5, 30, 12, 36, 11]
b = [2, 5, 6, 7, 10]
# using the true_divide and floor_divide functions to perform element-wise division
ans_true_divide = np.true_divide(a, b)
ans_floor_divide = np.floor_divide(a, b)

print("a =", a, "\nb =", b)
print("Result of true divide =", ans_true_divide)
print("Result of floor divide =", ans_floor_divide)

```

**输出:**

```py
a = [5, 30, 12, 36, 11] 
b = [2, 5, 6, 7, 10]
Result of true divide = [2.5        6\.         2\.         5.14285714 1.1       ]
Result of floor divide = [2 6 2 5 1]

```

这里输出的不同之处在于，在底除法的输出中，实际商的底作为输出呈现，而真正的除法方法将实际商包括在输出中。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy true_divide** 方法，并使用该方法练习了不同类型的示例。

如果你想了解更多关于 NumPy 的知识，请随意浏览我们的 NumPy 教程。

* * *

## 参考

*   [NumPy true_divide 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html)