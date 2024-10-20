# NumPy one–完整指南

> 原文：<https://www.askpython.com/python-modules/numpy/numpy-ones>

你好，欢迎来到这个关于**Numpy one**的教程。在本教程中，我们将学习 **NumPy ones()** 方法，也将看到许多关于相同的例子。让我们开始吧！

* * *

## NumPy ones 方法是什么？

NumPy `ones`返回给定形状和数据类型的一个 [**Numpy 数组**](https://www.askpython.com/python-modules/numpy/python-numpy-arrays) ，所有值都设置为 1。

* * *

## NumPy ones 的语法

让我们先来看看 NumPy ones 的语法。

```py
numpy.ones(shape, dtype=None, order='C', like=None)

```

| **参数** | **描述** | **必需/可选** |
| 形状 | 所需的数组形状。它可以是一个整数，也可以是一组整数。 | 需要 |
| 类型 | 所需的数组数据类型。
默认数据类型为**浮点型**。 | 可选择的 |
| 命令 | 多维数据在存储器中存储的期望顺序。它可以是主要行(' C ')或主要列(' F ')。
默认顺序为**【C】**，即**行主**。 | 可选择的 |
| like (array_like) | 引用对象，以允许创建不是 NumPy 数组的数组。 | 可选择的 |

**返回:**
给定形状、数据类型和顺序的数组，只填充 1。

* * *

## 使用 NumPy 的例子

现在让我们看一些`numpy.ones()`方法的实际例子。

### 使用 1 的一维数组

```py
import numpy as np

one_dim_array = np.ones(4)
print(one_dim_array) 

```

**输出:**

```py
[1\. 1\. 1\. 1.]

```

* * *

### 使用零的二维数组

**N×M 阵列**

```py
import numpy as np

two_dim_array = np.ones((4, 2))
print(two_dim_array) 

```

**输出:**

```py
[[1\. 1.]
 [1\. 1.]
 [1\. 1.]
 [1\. 1.]]

```

**1×N 阵列**

```py
import numpy as np

one_row_array = np.ones((1, 3))
print(one_row_array) 

```

**输出:**

```py
[[1\. 1\. 1.]]

```

**N×1 阵列**

```py
import numpy as np

one_col_array = np.ones((5, 1))
print(one_col_array) 

```

**输出:**

```py
[[1.]
 [1.]
 [1.]
 [1.]
 [1.]]

```

* * *

### 一维整型数组

```py
import numpy as np

one_dim_int_array = np.ones(5, dtype=np.int64)
print(one_dim_int_array) 

```

**输出:**

```py
[1 1 1 1 1]

```

* * *

### 二维整型数组

```py
import numpy as np

two_dim_int_array = np.ones((3, 3), dtype=np.int64)
print(two_dim_int_array) 

```

**输出:**

```py
[[1 1 1]
 [1 1 1]
 [1 1 1]]

```

* * *

### 一维自定义数据类型数组

```py
import numpy as np

custom_one_dim_array = np.ones(4, dtype=[('x', 'int'), ('y', 'float')])
print(custom_one_dim_array) 
print(custom_one_dim_array.dtype) 

```

**输出:**

```py
[(1, 1.) (1, 1.) (1, 1.) (1, 1.)]
[('x', '<i4'), ('y', '<f8')]

```

在这个例子中，我们将第一个值指定为 int，将第二个值指定为 float。

* * *

### 二维自定义数据类型数组

我们可以将数组的元素指定为元组，还可以指定它们的数据类型。

```py
import numpy as np

custom_two_dim_array = np.ones((2, 3), dtype=[('x', 'float'), ('y', 'int')])
print(custom_two_dim_array) 
print(custom_two_dim_array.dtype) 

```

**输出:**

```py
[[(1., 1) (1., 1) (1., 1)]
 [(1., 1) (1., 1) (1., 1)]]
[('x', '<f8'), ('y', '<i4')]

```

这里，代码将数组元素中元组的第一个值指定为 float，第二个值指定为 int。

* * *

## 结论

仅此而已！在本教程中，我们学习了 **Numpy ones** 方法，并使用相同的方法练习了不同类型的示例。
如果你想了解更多关于 NumPy 的信息，请随意浏览我们的 [NumPy 教程](https://www.askpython.com/python-modules/numpy)。

* * *

## 参考

*   [NumPy one 官方文档](https://numpy.org/doc/stable/reference/generated/numpy.ones.html)