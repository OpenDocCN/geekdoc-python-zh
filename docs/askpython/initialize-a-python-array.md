# 三种初始化 Python 数组的方法

> 原文：<https://www.askpython.com/python/array/initialize-a-python-array>

嘿，伙计们！在本文中，我们将关注初始化 Python 数组的一些**简单方法。**

* * *

## 什么是 Python 数组？

**Python 数组**是一种数据结构，在连续的内存位置保存相似的数据值。

与[列表](https://www.askpython.com/python/list/python-list)(动态数组)相比，Python 数组在其中存储了相似类型的元素。而 Python 列表可以存储属于不同数据类型的元素。

现在，让我们看看在 Python 中初始化数组的不同方法。

* * *

## 方法 1:使用 for 循环和 Python range()函数

[Python for 循环](https://www.askpython.com/python/python-for-loop)和 range()函数一起可以用来用默认值初始化一个数组。

**语法:**

```py
[value for element in range(num)]

```

[Python range()函数](https://www.askpython.com/python/built-in-methods/python-range-method)接受一个数字作为参数，返回一个数字序列，从 0 开始，到指定的数字结束，每次递增 1。

Python for loop 会将数组中位于 range()函数中指定的范围之间的每个元素的值设置为 0(默认值)。

**举例:**

```py
arr=[]
arr = [0 for i in range(5)] 
print(arr)

```

我们创建了一个数组“arr ”,并用带有默认值(0)的 5 个元素初始化它。

**输出:**

```py
[0, 0, 0, 0, 0]

```

* * *

## 方法 2: Python NumPy 模块创建并初始化数组

[Python NumPy 模块](https://www.askpython.com/python-modules/numpy/python-numpy-arrays)可以用来创建数组并高效地操作数组中的数据。numpy.empty()函数创建一个指定大小的数组，默认值为“None”。

**语法:**

```py
numpy.empty(size,dtype=object)

```

**举例:**

```py
import numpy as np
arr = np.empty(10, dtype=object) 
print(arr)

```

**输出:**

```py
[None None None None None None None None None None]

```

* * *

## 方法 3:初始化 Python 数组的直接方法

声明数组时，我们可以使用下面的命令初始化数据值:

```py
array-name = [default-value]*size

```

**举例:**

```py
arr_num = [0] * 5
print(arr_num)

arr_str = ['P'] * 10
print(arr_str)

```

如上面的例子所示，我们已经创建了两个数组，默认值为“0”和“P ”,以及指定的大小。

**输出:**

```py
[0, 0, 0, 0, 0]
['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P']

```

* * *

## 结论

到此，我们就结束了这个话题。如果你有任何疑问，请随时在下面评论。

* * *

## 参考

*   [Python 数组初始化—文档](https://numpy.org/doc/stable/reference/routines.array-creation.html)