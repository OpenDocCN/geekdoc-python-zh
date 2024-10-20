# NumPy 产品-完整指南

> 原文:# t0]https://www . aspython . com/python-modules/num py/numpy-prod

你好，欢迎来到 Numpy prod 教程。在本教程中，我们将学习 NumPy prod()方法，也将看到许多关于该方法的例子。让我们开始吧！

***也读作:[【NumPy 零点——完全指南】](https://www.askpython.com/python-modules/numpy/numpy-zeros)***

* * *

## 什么是 NumPy prod？

NumPy 中的 prod 方法是一个返回数组元素乘积的函数。它可以是所有数组元素的乘积、沿行数组元素的乘积或沿列数组元素的乘积。在本教程接下来的部分，我们将会看到每个例子。

* * *

## NumPy 产品的语法

让我们首先看看 NumPy prod 函数的语法。

```py
numpy.prod(a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)

```

| **参数** | **描述** | **必需/可选** |
| (类似数组) | 输入数据。 | 需要 |
| 轴 | 沿其计算数组乘积的轴。它可以是 axis=0，即沿列，也可以是 axis=1，即沿行，或者 axis=None，这意味着要返回整个数组的乘积。如果 axis 是一个整数元组，则乘积在元组中指定的所有轴上执行，而不是像以前那样在单个轴或所有轴上执行。 | 可选择的 |
| 数据类型 | 要返回的数组的数据类型。 | 可选择的 |
| 在外 | 放置结果的替代输出数组。它必须具有与预期输出相同的形状。 | 可选择的 |
| keepdims (bool) | 如果设置为真，减少的轴将作为尺寸为 1 的尺寸留在结果中。使用此选项，结果将根据输入数组正确传播。 | 可选择的 |
| 最初的 | 产品的起始值。 | 可选择的 |
| 在哪里 | 产品中包含的元素。 | 可选择的 |

**返回:**
一个与 *a* 形状相同的数组，包含 *a* 沿给定轴的元素的乘积，并去掉指定轴。如果 axis=None，则返回一个标量，它是整个数组的乘积。

* * *

## numpy.prod()的示例

让我们进入使用 numpy.prod()函数的不同例子。

### 整个数组的乘积

**一维数组**

```py
import numpy as np

a = [5, 3, 1]

product = np.prod(a)
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [5, 3, 1]
Product of the array = 15

```

数组的乘积= 5*3*1 = 15

**二维数组**

```py
import numpy as np

a = [[5, 3, 1], [1, 2, 4]]

product = np.prod(a)
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [[5, 3, 1], [1, 2, 4]]
Product of the array = 120

```

数组的乘积= 5*3*1*1*2*4 = 120

* * *

### 沿着轴的产品

**列式产品**

```py
import numpy as np

a = [[5, 3, 1], 
     [1, 2, 4]]

# product along axis=0 i.e. columns
product = np.sum(a, axis=0)
print("a =", a)
print("Product of the array along the columns =", product)

```

**输出:**

```py
a = [[5, 3, 1], [1, 2, 4]]
Product of the array along the columns = [5 6 4]

```

第 0 列乘积= 5*1 = 5
第 1 列乘积= 3*2 = 6
第 2 列乘积= 1*4 = 4

**逐行乘积**

```py
import numpy as np

a = [[5, 3, 1], 
     [1, 2, 4]]

# product along axis=1 i.e. rows
product = np.prod(a, axis=1)
print("a =", a)
print("Product of the array along the rows =", product)

```

**输出:**

```py
a = [[5, 3, 1], [1, 2, 4]]
Product of the array along the rows = [15  8]

```

第 0 行产品= 5*3*1 = 15
第 1 行产品= 1*2*4 = 8

* * *

### 以浮点数据类型返回数组的乘积

这与上面的例子相同，只是这里返回值是浮点数据类型。

**整个数组的乘积**

```py
import numpy as np

a = [[2, 3 ,6], 
     [1, 5, 4]]

product = np.prod(a, dtype=float)
print("a =", a)
print("Product of the array along the columns =", product)

```

**输出:**

```py
a = [[2, 3, 6], [1, 5, 4]]
Product of the array along the columns = 720.0

```

**列式产品**

```py
import numpy as np

a = [[2, 3 ,6], 
     [1, 5, 4]]

# product along axis=0 i.e. columns
product = np.prod(a, axis=0, dtype=float)
print("a =", a)
print("Product of the array along the columns =", product)

```

**输出:**

```py
a = [[2, 3, 6], [1, 5, 4]]
Product of the array along the columns = [ 2\. 15\. 24.]

```

**逐行乘积**

```py
import numpy as np

a = [[2, 3 ,6], 
     [1, 5, 4]]

# product along axis=1 i.e. rows
product = np.prod(a, axis=1, dtype=float)
print("a =", a)
print("Product of the array along the rows =", product)

```

**输出:**

```py
a = [[2, 3, 6], [1, 5, 4]]
Product of the array along the rows = [36\. 20.]

```

* * *

### 数组中特定元素的乘积

**一维数组**

```py
import numpy as np

a = [2, 9, 3, 4, 1]

product = np.prod(a, where=[True, False, False, True, True])
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [2, 9, 3, 4, 1]
Product of the array = 8

```

在上面的代码中， **'where'** 子句指定产品中包含哪些元素。“True”表示将该值包括在产品中,“False”表示不将该值包括在产品计算中。
这里，其中=[真，假，假，真，真]表示在乘积中只包括数组位置 0，3，4 的元素。因此，乘积= a[0]*a[3]*a[4] = 2*4*1 = 8。

**二维数组**

```py
import numpy as np

a = [[2, 9], 
    [7, 10]]

product = np.prod(a, where=[[True, False], [False, True]])
print("a =", a)
print("Product of the array =", product)

```

**输出:**

```py
a = [[2, 9], [7, 10]]
Product of the array = 20

```

这里，从第 0 行开始，只有 a[0][0]即 2，从第 1 行开始，只有 a[1][1]即 10 包括在乘积中。因此，乘积= 2*10 = 20。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy prod** 方法，并使用相同的方法练习了不同类型的示例。

* * *

## 参考

*   [NumPy 产品官方文档](https://numpy.org/doc/stable/reference/generated/numpy.prod.html#numpy.prod)